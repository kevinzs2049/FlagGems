"""
FlagGems ARM backend: Triton-CPU INT8 GEMM for quantized::linear_dynamic.

Replaces the OneDNN/ACL implementation of torch.ops.quantized.linear_dynamic
with a Triton-CPU i8mm kernel on ARM64 (SVE2 + i8mm).

Kernel configs (validated on CIX P1 CD8180, OMP=8):
  M=1        → BM=1,  BN=64, BK=4  (ConvertDotGeneric, 69 GOPS gate/up)
  M%64==0    → BM=64, BN=64, BK=32 (SVE2 i8mm, prefill path)
  otherwise  → BM=1,  BN=64, BK=4  (fallback)

E2E Qwen2-7B INT8 decode (OMP=8): 2.55 tok/s vs OneDNN 3.69 tok/s (0.69x).

Weight cache note: torch.library dispatch creates a new Python wrapper for
W_prepack on every call, so id(W_prepack) is not stable. We key the cache
on w.data_ptr() (physical address of the underlying weight storage) which
is stable for the lifetime of the model.
"""

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Triton INT8 GEMM kernel
# ---------------------------------------------------------------------------

@triton.jit
def _i8mm_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Tiled INT8 GEMM: C[M,N] = A[M,K] @ B[K,N], output int32."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k = k * BLOCK_K + tl.arange(0, BLOCK_K)
        a = tl.load(a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b = tl.load(b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
        acc += tl.dot(a, b)
    tl.store(
        c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
        acc.to(tl.int32),
    )


# ---------------------------------------------------------------------------
# Weight cache
# ---------------------------------------------------------------------------

# w_raw.data_ptr() → (weight_kn [K,N] int8, weight_scale float, bias or None)
#
# torch.library dispatch creates a NEW Python wrapper for W_prepack on every call,
# so id(W_prepack) changes each time and cannot be used as a cache key.
# Instead, we key on the data_ptr() of the raw qint8 weight tensor returned by
# unpack(): this is the actual memory address of the weight storage, which is
# stable for the lifetime of the model (weights don't move in memory).
_weight_cache: dict = {}


def _get_weight(W_prepack):
    w, bias = W_prepack.unpack()                       # w: qint8 [N, K] (cheap view)
    key = w.data_ptr()                                 # stable: physical address of weight data
    if key in _weight_cache:
        return _weight_cache[key]

    weight_kn = w.int_repr().T.contiguous()            # int8  [K, N]  (one-time transpose)
    weight_scale = float(w.q_scale())
    entry = (weight_kn, weight_scale, bias)
    _weight_cache[key] = entry
    return entry


# ---------------------------------------------------------------------------
# Core implementation
# ---------------------------------------------------------------------------

def _triton_quantized_linear_dynamic(X, W_prepack, reduce_range=False):
    """
    Triton-CPU replacement for torch.ops.quantized.linear_dynamic (CPU).

    X        : float32 tensor, shape [..., K]
    W_prepack: torch.ScriptObject (LinearPackedParamsBase), qint8 [N, K]
    Returns  : float32 tensor, shape [..., N]
    """
    weight_kn, weight_scale, bias = _get_weight(W_prepack)

    K = X.shape[-1]
    N = weight_kn.shape[1]
    orig_shape = X.shape

    # Flatten all leading dims into M
    x2d = X.view(-1, K)
    M = x2d.shape[0]

    # Dynamic quantization: find activation range → scale → int8
    x_abs_max = x2d.abs().max().item()
    if x_abs_max == 0.0:
        out2d = torch.zeros(M, N, dtype=torch.float32)
        if bias is not None:
            out2d = out2d + bias
        return out2d.view(*orig_shape[:-1], N)

    x_scale = x_abs_max / 127.0
    x_q = (x2d / x_scale).round_().clamp_(-128, 127).to(torch.int8)

    # Select BLOCK config based on M
    if M == 1:
        # Decode path: BK=4 → ConvertDotGeneric, LLVM fully unrolls K=4
        BM, BN, BK = 1, 64, 4
        grid = (1, N // BN)
    elif M % 64 == 0:
        # Prefill path: BM=64 BK=32 → SVE2 i8mm (alloca-hoist fix required)
        BM, BN, BK = 64, 64, 32
        grid = (M // BM, N // BN)
    else:
        # Fallback: BM=1 BK=4 works for any M (grid launches M blocks)
        BM, BN, BK = 1, 64, 4
        grid = (M, N // BN)

    c = torch.empty(M, N, dtype=torch.int32)
    _i8mm_kernel[grid](
        x_q, weight_kn, c, M, N, K,
        x_q.stride(0), x_q.stride(1),
        weight_kn.stride(0), weight_kn.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK,
    )

    # Dequantize: int32 → float32
    out2d = c.to(torch.float32).mul_(x_scale * weight_scale)
    if bias is not None:
        out2d = out2d + bias
    return out2d.view(*orig_shape[:-1], N)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

_quantized_lib = None   # keep reference alive to prevent GC


def register():
    """
    Register Triton implementation for quantized::linear_dynamic on CPU.

    Called automatically when the ARM backend is loaded (import flag_gems).
    Idempotent: safe to call multiple times.
    """
    global _quantized_lib
    if _quantized_lib is not None:
        return

    try:
        _quantized_lib = torch.library.Library("quantized", "IMPL")
        _quantized_lib.impl(
            "linear_dynamic",
            _triton_quantized_linear_dynamic,
            "CPU",
            allow_override=True,
        )
        logger.debug(
            "FlagGems ARM: registered Triton-CPU i8mm for quantized::linear_dynamic"
        )
    except Exception as e:
        logger.warning(
            f"FlagGems ARM: failed to register quantized::linear_dynamic override: {e}"
        )
