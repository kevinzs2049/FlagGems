"""Monkey-patch vLLM RMSNorm to use FlagGems fused Triton-CPU kernels.

Call `patch_vllm_rmsnorm()` after `flag_gems.enable()` and after vLLM
imports to replace the 5-op decomposed RMSNorm with a single fused kernel.
"""

import logging

import torch

from .fused_add_rms_norm import fused_add_rms_norm, rms_norm_forward

logger = logging.getLogger(__name__)

_PATCHED = False


def _forward_cpu_fused(self, x, residual=None):
    """Fused RMSNorm forward for CPU using Triton kernel."""
    if self.variance_size_override is not None:
        return self.forward_native(x, residual)

    weight = self.weight.data if self.has_weight else None
    if weight is None:
        return self.forward_native(x, residual)

    if residual is not None:
        normed, new_residual = fused_add_rms_norm(
            x, residual, [self.hidden_size], weight, self.variance_epsilon
        )
        return normed, new_residual
    else:
        return rms_norm_forward(
            x, [self.hidden_size], weight, self.variance_epsilon
        )


def patch_vllm_rmsnorm():
    """Patch vLLM RMSNorm.forward_cpu to use fused Triton kernels."""
    global _PATCHED
    if _PATCHED:
        return

    try:
        from vllm.model_executor.layers.layernorm import RMSNorm
        RMSNorm.forward_cpu = _forward_cpu_fused
        _PATCHED = True
        logger.info("Patched vLLM RMSNorm.forward_cpu with fused Triton kernel")
    except ImportError:
        logger.debug("vLLM not found, skipping RMSNorm patch")
    except Exception:
        logger.debug("Failed to patch vLLM RMSNorm", exc_info=True)
