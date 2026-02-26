"""Monkey-patch Qwen3RMSNorm (and compatible classes) to use the
fused ARM Triton-CPU RMSNorm kernel.

Qwen3RMSNorm uses a manual 5-op decomposition:
    to(fp32) → pow(2) → mean(-1) → rsqrt(var+eps) → mul → mul(weight)

This patch replaces that with a single fused kernel call that does all
the above in one pass (2-pass tiled to keep LLVM compile time short).

Affected modules (all share the same forward signature):
  - input_layernorm   N=hidden_size=1024  M=1  (decode)
  - post_attention_layernorm  N=1024  M=1
  - q_norm            N=head_dim=64       M=16  (16 heads, decode)
  - k_norm            N=head_dim=64       M=8   (8 KV-heads, decode)

Call patch_qwen3_rmsnorm() after loading the model and before the first
generate() call.
"""

import logging

from .fused_add_rms_norm import rms_norm_forward

logger = logging.getLogger(__name__)

_PATCHED_CLASSES: set = set()


def _make_triton_forward(cls_name: str):
    """Create a patched forward method that calls the fused Triton kernel."""

    def _forward(self, hidden_states):
        # rms_norm_forward handles:
        #   - internal fp32 accumulation (input is cast to fp32 in kernel)
        #   - arbitrary M (batch*seq*heads) and N (hidden_size or head_dim)
        #   - bfloat16 / float32 input and weight
        return rms_norm_forward(
            hidden_states,
            list(self.weight.shape),   # normalized_shape, e.g. [1024] or [64]
            self.weight,
            self.variance_epsilon,
        )

    _forward.__name__ = f"_triton_rms_norm_forward[{cls_name}]"
    return _forward


def patch_class(cls) -> bool:
    """Replace cls.forward with the fused kernel. Returns True if patched."""
    if cls in _PATCHED_CLASSES:
        return False
    # Minimal duck-type check: must have weight and variance_epsilon
    if not (hasattr(cls, "weight") or "weight" in getattr(cls, "__annotations__", {})):
        logger.debug("patch_class: %s has no 'weight', skipping", cls.__name__)
        return False
    cls._original_forward = cls.forward
    cls.forward = _make_triton_forward(cls.__name__)
    _PATCHED_CLASSES.add(cls)
    logger.info("Patched %s.forward with fused Triton RMSNorm", cls.__name__)
    return True


def patch_qwen3_rmsnorm() -> bool:
    """Patch Qwen3RMSNorm to use the fused ARM Triton-CPU kernel.

    Returns True if the patch was applied, False if already patched or
    if transformers is not available.
    """
    try:
        from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm
    except ImportError:
        logger.debug("transformers Qwen3RMSNorm not found, skipping patch")
        return False

    return patch_class(Qwen3RMSNorm)


def unpatch_qwen3_rmsnorm():
    """Restore the original Qwen3RMSNorm.forward (for testing)."""
    try:
        from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm
    except ImportError:
        return
    if Qwen3RMSNorm in _PATCHED_CLASSES:
        Qwen3RMSNorm.forward = Qwen3RMSNorm._original_forward
        _PATCHED_CLASSES.discard(Qwen3RMSNorm)
        logger.info("Restored original Qwen3RMSNorm.forward")
