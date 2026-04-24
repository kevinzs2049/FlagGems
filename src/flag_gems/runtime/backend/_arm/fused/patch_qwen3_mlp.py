"""Monkey-patch Qwen3MLP.forward to use triton-cpu's fused_mlp_bf16 kernel.

This patch replaces the 5-op ATen sequence
    gate_proj(x) → silu → up_proj(x) → mul → down_proj
with a single fused C kernel call when:
    - decode shape (M=1)
    - BF16 activation
    - gate_proj and up_proj are INT8 SDOT-packed Linears
      (expose attributes: _packed, _w_scale, K, N)

Measured benefit on Qwen3-1.7B W8A8-INT8 decode (3 rounds × 5 runs median,
CIX P1 CD8180, 8 big cores, OMP=8, performance governor):

  ENABLE_MLP_PATCH=1  ON  → 9.92 tok/s median (9.88, 10.04, 9.92)
  ENABLE_MLP_PATCH=0  OFF → 9.73 tok/s median (9.61, 9.73, 9.76)
  → +1.95% median (+2.5% mean) consistent across 3 rounds.

Usage:
    from flag_gems.runtime.backend._arm.fused.patch_qwen3_mlp import patch_qwen3_mlp
    patch_qwen3_mlp(model)
"""

import logging
import types

import torch

from triton.language.extra.cpu.runtime import fused_mlp_bf16

logger = logging.getLogger(__name__)

_PATCHED: set = set()


class FusedMLPWrapper:
    """Holds references to a Qwen3MLP's 3 projections + act_fn, exposes
    a forward that dispatches to the fused C kernel on M=1 BF16 decode.

    Falls back to composing gate/up/down via their own forward for:
      - M > 1 (prefill)
      - non-BF16 activation
      - gate/up not SDOT-packed INT8 Linears
    """

    def __init__(self, gate_linear, up_linear, down_linear, act_fn):
        self._gate_linear = gate_linear
        self._up_linear = up_linear
        self.down_proj = down_linear
        self.act_fn = act_fn

        self._fused = (
            hasattr(gate_linear, "_packed")
            and hasattr(up_linear, "_packed")
            and hasattr(gate_linear, "_w_scale")
            and hasattr(up_linear, "_w_scale")
            and hasattr(gate_linear, "K")
            and hasattr(gate_linear, "N")
        )
        if self._fused:
            self._gate_packed = gate_linear._packed
            self._up_packed = up_linear._packed
            self._gate_scale = gate_linear._w_scale
            self._up_scale = up_linear._w_scale
            self._K = gate_linear.K
            self._N = gate_linear.N

    def forward(self, x):
        shape = x.shape
        M = x.numel() // shape[-1]
        if self._fused and M == 1 and x.dtype == torch.bfloat16:
            xc = x.reshape(-1).contiguous()
            out = torch.empty(self._N, dtype=torch.bfloat16)
            fused_mlp_bf16(
                xc, self._gate_packed, self._up_packed,
                self._gate_scale, self._up_scale,
                out, self._K, self._N,
            )
            return self.down_proj(out.reshape(*shape[:-1], self._N))
        # ATen fallback: compose gate+up+silu+mul+down via each Linear's own forward
        gate = self._gate_linear(x)
        up = self._up_linear(x)
        return self.down_proj(self.act_fn(gate) * up)


def patch_qwen3_mlp(model) -> int:
    """Walk model, replace Qwen3MLP.forward with FusedMLPWrapper-based forward.

    Returns number of Qwen3MLP instances patched. Safe to call multiple times —
    each instance is patched once (tracked via id).
    """
    try:
        from transformers.models.qwen3.modeling_qwen3 import Qwen3MLP
    except ImportError:
        logger.debug("transformers Qwen3MLP not found, skipping patch")
        return 0

    n = 0
    for name, module in list(model.named_modules()):
        if isinstance(module, Qwen3MLP) and id(module) not in _PATCHED:
            wrapper = FusedMLPWrapper(
                module.gate_proj, module.up_proj,
                module.down_proj, module.act_fn,
            )
            module._original_forward = module.forward
            module._fused_mlp_wrapper = wrapper
            module.forward = types.MethodType(
                lambda self, x, _w=wrapper: _w.forward(x), module,
            )
            _PATCHED.add(id(module))
            n += 1
    if n > 0:
        logger.info("Patched %d Qwen3MLP.forward with fused_mlp_bf16", n)
    return n


def unpatch_qwen3_mlp(model) -> int:
    """Restore original Qwen3MLP.forward (for testing / revert)."""
    try:
        from transformers.models.qwen3.modeling_qwen3 import Qwen3MLP
    except ImportError:
        return 0
    n = 0
    for name, module in list(model.named_modules()):
        if isinstance(module, Qwen3MLP) and id(module) in _PATCHED:
            if hasattr(module, "_original_forward"):
                module.forward = module._original_forward
                del module._original_forward
                del module._fused_mlp_wrapper
            _PATCHED.discard(id(module))
            n += 1
    return n
