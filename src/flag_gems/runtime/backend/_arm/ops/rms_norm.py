"""
ARM CPU rms_norm / fused_add_rms_norm wrappers.

Wraps the existing _arm/fused/fused_add_rms_norm.py Triton kernels so they
can be used as drop-in replacements for flag_gems.rms_norm and
flag_gems.fused_add_rms_norm on ARM64 CPU.

Avoids the LibEntry/DEVICE_COUNT issue in the generic CUDA implementation
(flag_gems/ops/rms_norm.py indexes kernel_cache by GPU device count).

Benchmark (CIX P1 CD8180, OMP=8, BF16, H=2048):
  M=1:   ATen 44.6μs  Triton 26.5μs  → Triton +1.68x
  M=8:   ATen 71.1μs  Triton 30.4μs  → Triton +2.34x
  M=64:  ATen 394μs   Triton 74.5μs  → Triton +5.29x
  M=512: ATen 441μs   Triton 438μs   → Triton ~1.0x

ATen overhead at small M is OMP thread-pool synchronisation (8 threads for
tiny work); Triton-CPU avoids this via its own launch model.
Triton wins at all measured M values → no ATen fallback needed.
"""

from flag_gems.runtime.backend._arm.fused.fused_add_rms_norm import (
    fused_add_rms_norm as _arm_fused_add_rms_norm,
    rms_norm_forward as _arm_rms_norm_forward,
)


def rms_norm(x, normalized_shape, weight, eps=1e-5):
    """ARM CPU drop-in for flag_gems.rms_norm (→ ARM Triton kernel)."""
    return _arm_rms_norm_forward(x, normalized_shape, weight, eps)


def fused_add_rms_norm(x, residual, normalized_shape, weight, eps=1e-5):
    """
    ARM CPU drop-in for flag_gems.fused_add_rms_norm.

    In-place: residual = x + residual; x = rms_norm(residual) * weight.
    Returns (x, residual).
    """
    return _arm_fused_add_rms_norm(x, residual, normalized_shape, weight, eps)
