"""
ARM CPU fused silu_and_mul — pure ATen, no LibEntry.

flag_gems.silu_and_mul (fused/silu_and_mul.py) uses @pointwise_dynamic which
internally emits @libentry() → crashes on CPU-only ARM (DEVICE_COUNT=0).

Strategy: use F.silu(x1) * x2 (ATen NEON) for all shapes.

Benchmarks (CIX P1 CD8180, BF16, OMP=8, prefault+1000 runs, drop top-5%):
  Shape [1, 4096]  (decode):  ATen ~25μs  (Triton ~60μs — launch overhead)
  Shape [1, 6144]  (decode):  ATen ~36μs
  Shape [64, 4096] (prefill): ATen ~60μs  Triton ~65μs (similar; libsleef needed)

Conclusion: ATen NEON wins for all decode shapes; Triton offers no benefit even
for prefill at these sizes. No Triton used (avoids libsleef dependency).
"""

import torch
import torch.nn.functional as F


def arm_silu_and_mul(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """ARM CPU drop-in for flag_gems.silu_and_mul(x1, x2).

    Returns silu(x1) * x2 — uses ATen NEON (faster than Triton for all sizes).
    """
    return F.silu(x1) * x2


def arm_silu_and_mul_out(
    x1: torch.Tensor, x2: torch.Tensor, out: torch.Tensor
) -> torch.Tensor:
    """ARM CPU drop-in for flag_gems.silu_and_mul_out(x1, x2, out).

    Writes silu(x1) * x2 into pre-allocated 'out' tensor.
    """
    out.copy_(F.silu(x1) * x2)
    return out
