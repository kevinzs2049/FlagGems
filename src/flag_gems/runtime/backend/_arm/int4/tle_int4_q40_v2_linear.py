"""Q4_0 v2 nn.Linear replacement using llama.cpp-style packed layout.

Layout per (output row n, K-block-32 j):
  bytes 0..15: 32 nibbles. Q4_0 convention:
    byte b: low nibble = w[n, k=j*32+b]      (range [0..15], decoded as v - 8 → [-8..7])
    byte b: high nibble = w[n, k=j*32+b+16]
  bytes 16..17: fp16 weight scale for this (n, j) block.

So each block_q4_0 = 18 bytes. Per row: K/32 blocks × 18 bytes.
Total packed weight: N × K/32 × 18 bytes.

This matches llama.cpp's `block_q4_0` memory layout exactly, which lets the
inner SDOT loop process 32 K-elements per K-block (one uint8x16 load, two
SDOTs against int8 activation halves) — vs our prior layout that needed 8
SDOTs per K-block per (4-N tile).
"""
import torch
import triton
import triton.language as tl

from triton.language.extra.cpu.tle_ops import (
    sdot_gemv_q4_0_v2_bf16 as _cpu_q40_v2_gemv,
)


BLOCK_SIZE = 32


@triton.jit
def _tle_q40_v2_gemv_kernel(
    x_ptr, w_packed_ptr, out_ptr,
    K: tl.constexpr, N: tl.constexpr,
):
    _cpu_q40_v2_gemv(x_ptr, w_packed_ptr, out_ptr, K, N)


def quantize_w4_q4_0_v2(w_kn: torch.Tensor):
    """Q4_0-style per-block-32 quant of a [K, N] weight, returning packed bytes.

    The output values are stored as **unsigned nibbles in [0..15]** with
    decode = v - 8 (matches llama.cpp's Q4_0 convention). Allows SDOT to
    consume them after a single vsubq_s8.

    Returns:
        nibbles_kn:  [K, N] int8 with values in [0..15]
        block_scales_kn: [K/32, N] fp16
    """
    K, N = w_kn.shape
    assert K % 32 == 0 and N % 4 == 0, f"K%32==N%4==0 required; got K={K} N={N}"
    w_fp32 = w_kn.detach().to(torch.float32)
    w_blocks = w_fp32.reshape(K // 32, 32, N)            # [K/32, 32, N]
    absmax = w_blocks.abs().amax(dim=1).clamp(min=1e-8)  # [K/32, N]
    # Q4_0 uses range [-8, 7] (8 values negative, 7 positive). Scale = absmax/7
    # is conservative; llama.cpp uses absmax/-8 to actually reach the -8 endpoint.
    # We follow llama.cpp: scale = absmax / -8 (signed convention).
    # For symmetric usage, simpler: scale = absmax / 7 to keep range [-7, 7].
    # Use scale = absmax / 7 here for stability (we never produce -8).
    block_scales_fp32 = absmax / 7.0
    scales_b = block_scales_fp32.unsqueeze(1)             # [K/32, 1, N]
    # Quantize to signed [-7, 7], then offset to unsigned [1..15] = signed + 8
    w_int_signed = (w_blocks / scales_b).round().clamp(-7, 7).to(torch.int32)
    w_int_unsigned = (w_int_signed + 8).clamp(0, 15).to(torch.int8)  # [K/32, 32, N]
    nibbles_kn = w_int_unsigned.reshape(K, N).contiguous()
    block_scales_fp16 = block_scales_fp32.to(torch.float16).contiguous()
    return nibbles_kn, block_scales_fp16


def pack_q4_0_v2(w_nibbles_kn: torch.Tensor,
                  block_scales_kn: torch.Tensor) -> torch.Tensor:
    """Pack [K, N] unsigned-nibble weights + [K/32, N] fp16 scales into the
    flat [N × K/32 × 18] int8 layout the kernel expects.
    """
    K, N = w_nibbles_kn.shape
    K_blocks = K // 32
    assert block_scales_kn.shape == (K_blocks, N)

    # Reshape to [K_blocks, 32, N], then split into low (k=0..15) and high (k=16..31)
    w = w_nibbles_kn.reshape(K_blocks, 32, N)
    w_lo = w[:, :16, :]    # [K_blocks, 16, N] — low nibbles (k=0..15 within block)
    w_hi = w[:, 16:, :]    # [K_blocks, 16, N] — high nibbles (k=16..31 within block)

    # Pack: byte b within block = (low_nibble: w_lo[..., b, :] & 0x0F)
    #                              | ((high: w_hi[..., b, :] & 0x0F) << 4)
    # Output shape: [K_blocks, 16, N] uint8
    lo_i32 = w_lo.to(torch.int32) & 0x0F
    hi_i32 = w_hi.to(torch.int32) & 0x0F
    bytes_packed = (lo_i32 | (hi_i32 << 4)).to(torch.int8)  # [K_blocks, 16, N]

    # Now want layout [N, K_blocks, 18] flat. Permute to [N, K_blocks, 16] for nibbles.
    bytes_n_first = bytes_packed.permute(2, 0, 1).contiguous()    # [N, K_blocks, 16]
    scales_n_first = block_scales_kn.permute(1, 0).contiguous()   # [N, K_blocks] fp16

    # Concatenate along block-dim: 16 bytes nibbles + 2 bytes fp16 scale = 18 bytes
    out = torch.empty(N, K_blocks, 18, dtype=torch.int8)
    out[:, :, :16] = bytes_n_first
    # fp16 -> 2 bytes view as int8
    scales_bytes = scales_n_first.contiguous().view(torch.int8).reshape(N, K_blocks, 2)
    out[:, :, 16:] = scales_bytes
    return out.contiguous()


def dequant_q4_0_v2(w_packed: torch.Tensor, K: int, N: int) -> torch.Tensor:
    """Inverse of pack_q4_0_v2: reconstruct a [K, N] bf16 weight from the
    flat packed buffer. Used by prefill / fallback paths.
    """
    K_blocks = K // 32
    assert w_packed.shape == (N, K_blocks, 18)
    nibbles = w_packed[:, :, :16]    # [N, K_blocks, 16] int8 (each byte = 2 nibbles)
    scales_bytes = w_packed[:, :, 16:].contiguous().view(torch.float16).reshape(N, K_blocks)

    nib_i32 = nibbles.to(torch.int32) & 0xFF  # treat as unsigned bytes
    lo = (nib_i32 & 0x0F)                     # k=0..15
    hi = (nib_i32 >> 4) & 0x0F                # k=16..31
    # Decoded signed values = nibble - 8
    lo_s = (lo - 8).to(torch.int8)            # [N, K_blocks, 16]
    hi_s = (hi - 8).to(torch.int8)
    # Stack to [N, K_blocks, 32]
    full = torch.cat([lo_s, hi_s], dim=-1).to(torch.float32)  # [N, K_blocks, 32]
    # Multiply by per-block scale
    full = full * scales_bytes.unsqueeze(-1).to(torch.float32)  # [N, K_blocks, 32]
    # Reshape to [N, K] then transpose to [K, N]
    w_nk = full.reshape(N, K)
    return w_nk.t().contiguous().to(torch.bfloat16)  # [K, N] bf16


class TLEInt4Q40V2Linear(torch.nn.Module):
    """nn.Linear replacement using Q4_0 v2 layout for decode SDOT GEMV +
    dequantize-then-mm prefill.
    """

    def __init__(self, w_int_unsigned_kn: torch.Tensor,
                  block_scales: torch.Tensor):
        super().__init__()
        K, N = w_int_unsigned_kn.shape
        if K % 32 != 0 or N % 4 != 0:
            raise ValueError(f"K%32==N%4==0 required, got K={K} N={N}")
        self.K, self.N = K, N
        # Pack into the [N, K_blocks, 18] flat layout
        self._packed = pack_q4_0_v2(w_int_unsigned_kn, block_scales)
        # Lazy-built bf16 dequant cache for prefill / fallback
        self._w_dequant_kn_bf16 = dequant_q4_0_v2(self._packed, K, N)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        M = x.numel() // shape[-1]
        if M == 1 and x.dtype == torch.bfloat16:
            xc = x.reshape(-1).contiguous()
            out = torch.empty(self.N, dtype=torch.bfloat16)
            _tle_q40_v2_gemv_kernel[(1,)](
                xc, self._packed, out, K=self.K, N=self.N,
            )
            return out.reshape(*shape[:-1], self.N)
        # Prefill / non-bf16: dequant → ATen mm
        x_2d = x.reshape(-1, self.K).contiguous()
        out_2d = x_2d.to(torch.bfloat16) @ self._w_dequant_kn_bf16
        return out_2d.reshape(*shape[:-1], self.N)

    def extra_repr(self) -> str:
        return f"in_features={self.K}, out_features={self.N}, dtype=int4_q4_0_v2"
