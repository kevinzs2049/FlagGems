"""Drop-in nn.Linear replacement using TLE W4A8 SDOT GEMV decode + i4→i8
unpack + torch._int_mm prefill.

Decode (M=1, BF16): bf16 activation → in-kernel quant → INT8 SDOT matmul
with on-the-fly W4 unpack → bf16 output, all in one @triton.jit that
calls triton-cpu's `sdot_gemv_w4a8_bf16` TLE builtin.

Prefill (M>1): on-the-fly i4→i8 weight unpack to a temporary [K, N] int8
buffer, then per-row dynamic int8 activation quant, then torch._int_mm
(SVE2 i8mm). The temporary unpacked-int8 weight is allocated per call —
prefill is a one-shot at the start of generation, so the alloc cost is
amortized.

Weight packing layout (per-channel symmetric W4, in [-7, 7]):
    _packed:  [K/4, N/4, 4, 2] int8  — 8 packed bytes per (kb, nb) block
    _w_scale: [N] fp32              — per-output-channel scale
"""

import torch
import triton
import triton.language as tl

from triton.language.extra.cpu.tle_ops import (
    sdot_gemv_w4a8_bf16 as _cpu_w4a8_gemv,
)


@triton.jit
def _tle_w4a8_gemv_kernel(
    x_ptr, b_packed_ptr, w_scale_ptr, out_ptr,
    K: tl.constexpr, N: tl.constexpr,
):
    """Decode W4A8 GEMV: BF16 x [K] @ packed-i4 W [K/4, N/4, 4, 2] → BF16 out [N]."""
    _cpu_w4a8_gemv(x_ptr, b_packed_ptr, w_scale_ptr, out_ptr, K, N)


def pack_w4_per_channel(w_int4_kn: torch.Tensor) -> torch.Tensor:
    """Pack [K, N] int4 weights (values in -7..7, stored as int8) into
    [K/4, N/4, 4, 2] int8 (each byte holds 2 i4 nibbles).

    Layout matches the C kernel sdot_gemv_w4_range / sdot_pack_weights_w4:
      For each (kb, nb) block, 8 packed bytes hold 16 i4 weights as
        byte (ni*2 + p): low nibble = w[ni, ki=2p], high nibble = w[ni, ki=2p+1]
      where ni = N-stripe (0..3), ki = K-stripe (0..3) within the 4×4 block.
    """
    K, N = w_int4_kn.shape
    if K % 4 != 0 or N % 4 != 0:
        raise ValueError(f"pack_w4_per_channel requires K%4==N%4==0, got K={K} N={N}")
    if w_int4_kn.dtype != torch.int8:
        raise TypeError(f"w_int4_kn must be int8 (storing i4 in -7..7), got {w_int4_kn.dtype}")

    # Reshape [K, N] → [K/4, 4, N/4, 4] (kb, ki, nb, ni)
    w = w_int4_kn.reshape(K // 4, 4, N // 4, 4)
    # Permute to [K/4, N/4, ni, ki]
    w = w.permute(0, 2, 3, 1).contiguous()  # [K/4, N/4, 4, 4]
    # Split ki into pairs (lo, hi) along last axis
    w_pairs = w.reshape(K // 4, N // 4, 4, 2, 2)
    lo = w_pairs[..., 0]  # [K/4, N/4, 4, 2]   — w[ki=0] and w[ki=2]
    hi = w_pairs[..., 1]  # [K/4, N/4, 4, 2]   — w[ki=1] and w[ki=3]
    # Pack: byte = (lo & 0x0F) | ((hi & 0x0F) << 4); cast to int8 preserves bit pattern
    packed = ((lo.to(torch.int32) & 0x0F)
              | ((hi.to(torch.int32) & 0x0F) << 4)).to(torch.int8)
    return packed.contiguous()


def unpack_w4_per_channel(w_packed: torch.Tensor, K: int, N: int) -> torch.Tensor:
    """Inverse of pack_w4_per_channel: [K/4, N/4, 4, 2] int8 → [K, N] int8 in -7..7.

    Used in the prefill path to reconstitute a regular [K, N] int8 weight matrix
    for torch._int_mm (which requires unpacked int8).
    """
    assert w_packed.shape == (K // 4, N // 4, 4, 2)
    pi = w_packed.to(torch.int32)
    # Sign-extend low nibble: ((pi << 4) >> 4) on int8 semantics — emulate in int32
    lo = ((pi & 0x0F) ^ 0x08) - 0x08   # 4-bit two's-complement
    hi = (((pi >> 4) & 0x0F) ^ 0x08) - 0x08
    # Combine to [K/4, N/4, ni=4, ki=4]
    pairs = torch.stack([lo, hi], dim=-1)  # [K/4, N/4, 4, 2, 2]  — last dim: (lo_or_hi)
    full = pairs.reshape(K // 4, N // 4, 4, 4)  # [K/4, N/4, ni, ki]
    full = full.permute(0, 3, 1, 2).contiguous().reshape(K, N).to(torch.int8)
    return full


class TLEInt4Linear(torch.nn.Module):
    """nn.Linear replacement with TLE W4A8 SDOT decode + i4→i8 unpack + _int_mm prefill.

    Args:
        w_int4: [N, K] int8 tensor (values in -7..7) — pre-quantized weight
        w_scale: [N] fp32 tensor (per-output-channel scale)

    Required: K % 4 == 0 and N % 4 == 0 (SDOT lane requirement).

    Decode forward (T=1, BF16) issues a single TLE GEMV call. Prefill (T>1)
    unpacks i4→i8 on-the-fly and routes to torch._int_mm.
    """

    def __init__(self, w_int4: torch.Tensor, w_scale: torch.Tensor,
                  eager_unpack: bool = True):
        super().__init__()
        if w_int4.dtype != torch.int8:
            raise TypeError(f"w_int4 must be int8 (storing i4), got {w_int4.dtype}")
        self.N, self.K = w_int4.shape
        # [K, N] int8 (same axis order as TLEInt8Linear stores _w_int8_kn)
        w_kn = w_int4.t().contiguous()
        # _packed: [K/4, N/4, 4, 2] int8 — for decode SDOT W4
        self._packed = pack_w4_per_channel(w_kn)
        self._w_scale = w_scale.squeeze().to(torch.float32).contiguous()
        # Cache of [K, N] int8 unpacked weight for prefill _int_mm.
        # Built eagerly by default — the lazy build is single-threaded torch
        # ops on large tensors, which dominates per-call latency in PPL /
        # multi-token forward eval. Costs ~K*N bytes per Linear (e.g. ~2 GB
        # extra on Qwen3-4B). Set eager_unpack=False to defer (ok for pure
        # decode workloads that never hit the prefill path).
        if eager_unpack:
            self._unpacked_kn = w_kn  # already int8 in [-7, 7], same memory
        else:
            self._unpacked_kn = None

    def _get_unpacked_kn(self) -> torch.Tensor:
        if self._unpacked_kn is None:
            self._unpacked_kn = unpack_w4_per_channel(
                self._packed, self.K, self.N)
        return self._unpacked_kn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        M = x.numel() // shape[-1]
        if M == 1 and x.dtype == torch.bfloat16:
            xc = x.reshape(-1).contiguous()
            out = torch.empty(self.N, dtype=torch.bfloat16)
            _tle_w4a8_gemv_kernel[(1,)](
                xc, self._packed, self._w_scale, out,
                K=self.K, N=self.N,
            )
            return out.reshape(*shape[:-1], self.N)

        # Prefill / non-bf16: dynamic per-row int8 activation quant +
        # torch._int_mm with i4→i8 unpacked weight.
        xf = x.reshape(-1, self.K).contiguous()
        xf32 = xf.float()
        absmax = xf32.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
        x_scale = absmax / 127.0
        x_int8 = (xf32 / x_scale).clamp_(-128, 127).to(torch.int8)
        w_int8_kn = self._get_unpacked_kn()
        try:
            out_i32 = torch._int_mm(x_int8, w_int8_kn)
            out_f32 = out_i32.float() * x_scale * self._w_scale.unsqueeze(0)
        except Exception:
            w_fp32 = w_int8_kn.to(torch.float32) * self._w_scale.unsqueeze(0)
            out_f32 = xf32 @ w_fp32
        return out_f32.to(torch.bfloat16).reshape(*shape[:-1], self.N)

    def extra_repr(self) -> str:
        return f"in_features={self.K}, out_features={self.N}, dtype=int4 (W4A8)"
