"""Monkey-patch Qwen3MLP / Qwen3_5MLP forward to call the fused
TLE fused_swiglu_q4_0_v2 kernel for the gate+up+silu+mul half of the MLP.

Replaces 5 ATen ops (gate_proj, up_proj, silu, mul, alloc) and 4 intermediate
tensors with a single dispatch — directly attacking the small-op-dispatch
overhead that profile shows accounts for ~45% of decode time. Down_proj
stays as the existing TLEInt4Q40V2Linear path (its own GEMV).

Fast path requirements:
  - x is BF16, last-dim contiguous, M=1 (decode only)
  - both gate_proj and up_proj are TLEInt4Q40V2Linear with same K, N
  - K % 32 == 0, N % 4 == 0

Fallback to original forward otherwise.
"""
import logging
import types

import torch
import triton
import triton.language as tl

from triton.language.extra.cpu.tle_ops import fused_swiglu_q4_0_v2 as _tle_swiglu

logger = logging.getLogger(__name__)

_PATCHED: set = set()


@triton.jit
def _swiglu_q4_0_v2_kernel(x_ptr, gate_packed_ptr, up_packed_ptr, out_ptr,
                            K: tl.constexpr, N: tl.constexpr):
    _tle_swiglu(x_ptr, gate_packed_ptr, up_packed_ptr, out_ptr, K, N)


def _is_q40v2_linear(mod):
    """Detect TLEInt4Q40V2Linear by attribute fingerprint."""
    return (hasattr(mod, "_packed") and hasattr(mod, "K")
            and hasattr(mod, "N") and not hasattr(mod, "_w_scale"))


def _patched_mlp_forward(self, x):
    if (x.dtype == torch.bfloat16
            and x.numel() // x.shape[-1] == 1
            and self._tle_fast_path):
        xc = x.reshape(-1).contiguous()
        K, N = self._tle_K, self._tle_N
        intermediate = torch.empty(N, dtype=torch.bfloat16)
        _swiglu_q4_0_v2_kernel[(1,)](
            xc, self._tle_gate_packed, self._tle_up_packed,
            intermediate, K=K, N=N,
        )
        # down_proj on the fused result, then reshape back
        out = self.down_proj(intermediate.reshape(*x.shape[:-1], N))
        return out

    return self._original_forward(x)


def _get_qwen_mlp_classes():
    classes = []
    for modname, clsname in [
        ("transformers.models.qwen3.modeling_qwen3", "Qwen3MLP"),
        ("transformers.models.qwen3_5.modeling_qwen3_5", "Qwen3_5MLP"),
        ("transformers.models.qwen3_5_moe.modeling_qwen3_5_moe", "Qwen3_5MLP"),
        ("transformers.models.qwen3_next.modeling_qwen3_next", "Qwen3NextMLP"),
    ]:
        try:
            mod = __import__(modname, fromlist=[clsname])
            classes.append(getattr(mod, clsname))
        except (ImportError, AttributeError):
            pass
    return tuple(classes)


def patch_qwen3_5_swiglu_q4_0_v2(model) -> int:
    mlp_classes = _get_qwen_mlp_classes()
    if not mlp_classes:
        return 0
    n = 0
    for _name, mod in list(model.named_modules()):
        if not isinstance(mod, mlp_classes) or id(mod) in _PATCHED:
            continue
        gate = getattr(mod, "gate_proj", None)
        up = getattr(mod, "up_proj", None)
        if gate is None or up is None:
            continue
        if not (_is_q40v2_linear(gate) and _is_q40v2_linear(up)):
            continue
        if gate.K != up.K or gate.N != up.N:
            continue
        if gate.K % 32 != 0 or gate.N % 4 != 0:
            continue
        mod._tle_K = gate.K
        mod._tle_N = gate.N
        mod._tle_gate_packed = gate._packed
        mod._tle_up_packed = up._packed
        mod._tle_fast_path = True
        mod._original_forward = mod.forward
        mod.forward = types.MethodType(_patched_mlp_forward, mod)
        _PATCHED.add(id(mod))
        n += 1
    if n > 0:
        logger.info(
            "Patched %d Qwen3/3.5 MLP modules with TLE fused_swiglu_q4_0_v2", n)
    return n


def unpatch_qwen3_5_swiglu_q4_0_v2(model) -> int:
    mlp_classes = _get_qwen_mlp_classes()
    if not mlp_classes:
        return 0
    n = 0
    for _name, mod in list(model.named_modules()):
        if isinstance(mod, mlp_classes) and id(mod) in _PATCHED:
            if hasattr(mod, "_original_forward"):
                mod.forward = mod._original_forward
                del mod._original_forward
                del mod._tle_K
                del mod._tle_N
                del mod._tle_gate_packed
                del mod._tle_up_packed
                del mod._tle_fast_path
            _PATCHED.discard(id(mod))
            n += 1
    return n
