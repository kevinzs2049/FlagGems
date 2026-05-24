"""Drop-in nn.Linear replacement using Q4_0-style W4A8 SDOT GEMV decode +
per-block-32 dequant prefill.

Q4_0 quantization layout (mirrors llama.cpp):
- Block size = 32 K-elements
- Per (output_row, K-block): 1 fp16 scale + 32 int4 weights
- Per-output-channel-per-block scale (NOT per-channel) — handles outliers
  far better than per-channel W4 by localizing the scale to small K-windows.

Decode (M=1, BF16): bf16 x → in-kernel int8 quant → SDOT with on-the-fly W4
unpack and per-block dequant (int32 × block_scale × x_scale → fp32 accum) →
bf16 output. One TLE op call.

Prefill (M>1): on-the-fly i4→i8 unpack and per-block-scale broadcast to
[K, N] fp32 weight matrix, then per-row int8 activation quant + ATen mm
(BF16) with the dequantized weights cast to bf16. Slower than the per-channel
W4 path's prefill (which can reuse `torch._int_mm`) but bounded by the
prefill being a one-shot.
"""

import torch
import triton
import triton.language as tl

from triton.language.extra.cpu.tle_ops import (
    sdot_gemv_q4_0_bf16 as _cpu_q40_gemv,
)


BLOCK_SIZE = 32


@triton.jit
def _tle_q40_gemv_kernel(
    x_ptr, b_packed_ptr, block_scales_ptr, out_ptr,
    K: tl.constexpr, N: tl.constexpr,
):
    """Q4_0 GEMV: bf16 x [K] @ packed-i4 W [K/4, N/4, 4, 2] + scales [K/32, N] → bf16 out [N]."""
    _cpu_q40_gemv(x_ptr, b_packed_ptr, block_scales_ptr, out_ptr, K, N)


def quantize_w4_q4_0_per_block(w_kn: torch.Tensor):
    """Q4_0-style quantize [K, N] fp/bf16 weight to per-block-32 W4.

    Returns:
        w_int4_kn:    [K, N] int8 with values in [-7, 7]   (canonical, before pack)
        block_scales: [K/32, N] fp16  (per-block-32 per-output-channel scale)
    """
    K, N = w_kn.shape
    assert K % 32 == 0 and N % 4 == 0
    w_fp32 = w_kn.detach().to(torch.float32)
    # Reshape to [K/32, 32, N], compute per-block per-output absmax along K-block
    w_blocks = w_fp32.reshape(K // 32, 32, N)
    absmax = w_blocks.abs().amax(dim=1).clamp(min=1e-8)  # [K/32, N]
    block_scales_fp32 = absmax / 7.0
    # Quantize: divide each block by its scale, round, clamp to [-7, 7]
    scales_broadcast = block_scales_fp32.unsqueeze(1)  # [K/32, 1, N]
    w_int4_blocks = (w_blocks / scales_broadcast).round().clamp(-7, 7).to(torch.int8)
    w_int4_kn = w_int4_blocks.reshape(K, N).contiguous()
    block_scales_fp16 = block_scales_fp32.to(torch.float16).contiguous()
    return w_int4_kn, block_scales_fp16


def pack_w4_q4_0(w_int4_kn: torch.Tensor) -> torch.Tensor:
    """Pack [K, N] int4-storage-as-int8 weights into [K/4, N/4, 4, 2] int8.
    Same byte layout as pack_w4_per_channel — just the meaning of the scales
    changes. We re-export under the q4_0 name for clarity.
    """
    K, N = w_int4_kn.shape
    if K % 4 != 0 or N % 4 != 0:
        raise ValueError(f"K%4==N%4==0 required, got K={K} N={N}")
    if K % 32 != 0:
        raise ValueError(f"K%32==0 required for Q4_0, got K={K}")
    w = w_int4_kn.reshape(K // 4, 4, N // 4, 4)
    w = w.permute(0, 2, 3, 1).contiguous()  # [K/4, N/4, 4, 4]
    w_pairs = w.reshape(K // 4, N // 4, 4, 2, 2)
    lo = w_pairs[..., 0]
    hi = w_pairs[..., 1]
    packed = ((lo.to(torch.int32) & 0x0F)
              | ((hi.to(torch.int32) & 0x0F) << 4)).to(torch.int8)
    return packed.contiguous()


def dequant_w4_q4_0_to_kn(w_int4_kn: torch.Tensor,
                           block_scales_fp16: torch.Tensor) -> torch.Tensor:
    """Reconstruct [K, N] fp32 weight from per-block-32 W4 quant.
    Used for prefill / fallback.
    """
    K, N = w_int4_kn.shape
    assert block_scales_fp16.shape == (K // 32, N)
    w_blocks = w_int4_kn.reshape(K // 32, 32, N).to(torch.float32)
    scales = block_scales_fp16.to(torch.float32).unsqueeze(1)  # [K/32, 1, N]
    return (w_blocks * scales).reshape(K, N)


class TLEInt4Q40Linear(torch.nn.Module):
    """nn.Linear replacement with Q4_0 W4A8 SDOT decode + dequant-then-mm prefill.

    Args:
        w_int4_kn: [K, N] int8 (values in -7..7) — pre-quantized weight in
                   K-major order
        block_scales: [K/32, N] fp16 per-block-32 per-output-channel scale

    Required: K % 32 == 0 and N % 4 == 0.

    Decode (T=1) issues one TLE Q4_0 GEMV call. Prefill (T>1) reconstructs a
    bf16 [K, N] weight via dequant_w4_q4_0_to_kn then runs F.linear. The
    bf16 weight cache is built lazily on first prefill call.
    """

    def __init__(self, w_int4_kn: torch.Tensor, block_scales: torch.Tensor):
        super().__init__()
        if w_int4_kn.dtype != torch.int8:
            raise TypeError(f"w_int4_kn must be int8, got {w_int4_kn.dtype}")
        K, N = w_int4_kn.shape
        if K % 32 != 0 or N % 4 != 0:
            raise ValueError(f"K%32==N%4==0 required, got K={K} N={N}")
        if block_scales.shape != (K // 32, N):
            raise ValueError(
                f"block_scales must be [K/32, N] = [{K//32}, {N}], "
                f"got {tuple(block_scales.shape)}")
        self.K = K
        self.N = N
        self._packed = pack_w4_q4_0(w_int4_kn)               # [K/4, N/4, 4, 2] int8
        self._block_scales = block_scales.to(torch.float16).contiguous()  # [K/32, N] fp16
        # Lazy cache of dequantized [K, N] bf16 weight, used by prefill.
        self._w_dequant_kn_bf16 = None
        # Keep w_int4 around for cheap eager dequant on first prefill call.
        self._w_int4_kn = w_int4_kn

    def _get_dequant_kn(self) -> torch.Tensor:
        if self._w_dequant_kn_bf16 is None:
            self._w_dequant_kn_bf16 = dequant_w4_q4_0_to_kn(
                self._w_int4_kn, self._block_scales).to(torch.bfloat16).contiguous()
            # Free the int4 view since the bf16 dequant covers prefill needs.
            self._w_int4_kn = None
        return self._w_dequant_kn_bf16

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        M = x.numel() // shape[-1]
        if M == 1 and x.dtype == torch.bfloat16:
            xc = x.reshape(-1).contiguous()
            out = torch.empty(self.N, dtype=torch.bfloat16)
            _tle_q40_gemv_kernel[(1,)](
                xc, self._packed, self._block_scales, out,
                K=self.K, N=self.N,
            )
            return out.reshape(*shape[:-1], self.N)

        # Prefill / non-bf16: dequantize to bf16 and use F.linear.
        w_kn = self._get_dequant_kn()  # [K, N] bf16
        # F.linear expects [N, K] weight; we have [K, N] dequant matching x @ w.
        x_2d = x.reshape(-1, self.K).contiguous()
        out_2d = x_2d.to(torch.bfloat16) @ w_kn  # [M, N] bf16
        return out_2d.reshape(*shape[:-1], self.N)

    def extra_repr(self) -> str:
        return f"in_features={self.K}, out_features={self.N}, dtype=int4_q4_0 (per-block-32)"
