"""
FlagGems ARM backend: Triton-CPU INT8 GEMM for quantized::linear_dynamic.

Replaces the OneDNN/ACL implementation of torch.ops.quantized.linear_dynamic
with a Triton-CPU i8mm kernel on ARM64 (SVE2 + i8mm).

Kernel configs (validated on CIX P1 CD8180):
  M=1          → BM=1,  BN=64, BK=4  (ConvertDotGeneric, 69 GOPS decode)
  M=2          → BM=2,  BN=64, BK=4  (ConvertDotGeneric, LLVM unrolls K=4)
  M%64==0      → BM=64, BN=64, BK=32 (SVE2 i8mm dynamic ForOp, 411 GOPS)
  M%8==0       → BM=8,  BN=64, BK=32 (SVE2 i8mm dynamic ForOp)
  M%4==0       → BM=4,  BN=64, BK=32 (SVE2 i8mm static path)
  otherwise    → BM=1,  BN=64, BK=4  (fallback for M=3,5,6,7...)

Fusion optimisation (2026-03-06):
  _i8mm_fused_kernel takes FP32 activation input directly and outputs FP32.
  Quantisation (FP32→INT8) and dequantisation (INT32→FP32) are fused inside
  the kernel, eliminating 7 separate PyTorch operator calls per linear layer:
    BEFORE: abs, max, div, round_, clamp_, to(int8), empty(int32),
            dot-kernel, to(float32), mul_
    AFTER:  abs, max,  fused-kernel  (saves ~17 ms/tok on Qwen3-1.7B)

Weight cache: keyed on w.data_ptr() (stable physical address).
"""

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Fused kernel: FP32 input → INT8 quant → INT8 GEMM → FP32 dequant output
# ---------------------------------------------------------------------------

@triton.jit
def _i8mm_fused_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    inv_x_scale,          # float32 scalar: 127.0 / x_abs_max
    out_scale,            # float32 scalar: (x_abs_max / 127.0) * w_scale
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Fused INT8 GEMM:
      A[M,K] fp32 → quantise in-kernel → INT8
      B[K,N] int8  (pre-transposed weight)
      C[M,N] fp32  = dequant(A_q @ B) stored directly

    Eliminates external quant/dequant PyTorch ops and the int32 accumulator
    buffer.  The SVE2 i8mm (smmla) path is preserved: after in-kernel cast
    both operands are int8, so ConvertDotToSVE2I8MM still fires for BK=32.
    For BK=4 (M=1 decode), ConvertDotGeneric with LLVM full-unroll is used.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k = k * BLOCK_K + tl.arange(0, BLOCK_K)

        # Load FP32 activation tile; quantise to INT8 in-kernel
        a_fp32 = tl.load(
            a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        )
        # Scale, clamp [-128, 127], truncate to int8
        # tl.minimum/maximum = elementwise, no libdevice needed
        a_scaled = a_fp32 * inv_x_scale
        a_clamped = tl.minimum(tl.maximum(a_scaled, -128.0), 127.0)
        a_int8 = a_clamped.to(tl.int8)

        b = tl.load(
            b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
        )
        acc += tl.dot(a_int8, b)

    # Dequantise: int32 → float32, scale and store
    c_fp32 = acc.to(tl.float32) * out_scale
    tl.store(
        c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
        c_fp32,
    )


# ---------------------------------------------------------------------------
# Legacy unfused kernel (kept for reference / debugging)
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
    """Unfused INT8 GEMM: A int8, B int8 → C int32."""
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
_weight_cache: dict = {}


def _get_weight(W_prepack):
    w, bias = W_prepack.unpack()                       # w: qint8 [N, K] (cheap view)
    key = w.data_ptr()                                 # stable physical address
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

    Uses _i8mm_fused_kernel: FP32 activation → (quantise in-kernel) →
    INT8 GEMM → (dequantise in-kernel) → FP32 output.
    Eliminates 7 external PyTorch ops vs the original unfused path.
    """
    weight_kn, weight_scale, bias = _get_weight(W_prepack)

    K = X.shape[-1]
    N = weight_kn.shape[1]
    orig_shape = X.shape

    x2d = X.view(-1, K)
    M = x2d.shape[0]

    # Compute activation scale (one reduction, unavoidable)
    x_abs_max = x2d.abs().max().item()
    if x_abs_max == 0.0:
        out2d = torch.zeros(M, N, dtype=torch.float32)
        if bias is not None:
            out2d = out2d + bias
        return out2d.view(*orig_shape[:-1], N)

    # Scalars for fused kernel (no separate x_q buffer, no int32 output buffer)
    inv_x_scale = 127.0 / x_abs_max
    out_scale   = (x_abs_max / 127.0) * weight_scale

    # Select BLOCK config based on M alignment.
    # SVE2 i8mm (ConvertDotToSVE2I8MM) requires: M==2 or M%4==0, N%4==0, K%8==0.
    # Dynamic ForOp path additionally requires: K%16==0, M%8==0.
    if M == 1:
        BM, BN, BK = 1, 64, 4
        grid = (1, N // BN)
    elif M == 2:
        BM, BN, BK = 2, 64, 4
        grid = (1, N // BN)
    elif M % 64 == 0:
        BM, BN, BK = 64, 64, 32
        grid = (M // 64, N // BN)
    elif M % 8 == 0:
        BM, BN, BK = 8, 64, 32
        grid = (M // 8, N // BN)
    elif M % 4 == 0:
        BM, BN, BK = 4, 64, 32
        grid = (M // 4, N // BN)
    else:
        BM, BN, BK = 1, 64, 4
        grid = (M, N // BN)

    out2d = torch.empty(M, N, dtype=torch.float32)
    _i8mm_fused_kernel[grid](
        x2d, weight_kn, out2d,
        M, N, K,
        x2d.stride(0), x2d.stride(1),
        weight_kn.stride(0), weight_kn.stride(1),
        out2d.stride(0), out2d.stride(1),
        inv_x_scale=inv_x_scale,
        out_scale=out_scale,
        BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK,
    )

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
            "FlagGems ARM: registered Triton-CPU i8mm (fused) for quantized::linear_dynamic"
        )
    except Exception as e:
        logger.warning(
            f"FlagGems ARM: failed to register quantized::linear_dynamic override: {e}"
        )
