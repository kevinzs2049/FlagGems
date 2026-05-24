import torch

import flag_gems


def _patch_rmsnorm_class(cls):
    if getattr(cls, "__flag_gems_triton_rmsnorm_patched__", False):
        return False

    def _forward(self, hidden_states):
        if (
            hidden_states.device.type == "cpu"
            and self.weight.device == hidden_states.device
            and self.weight.dtype == hidden_states.dtype
        ):
            return flag_gems.rms_norm(
                hidden_states,
                [hidden_states.shape[-1]],
                self.weight,
                self.variance_epsilon,
            )

        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    cls.forward = _forward
    cls.__flag_gems_triton_rmsnorm_patched__ = True
    return True


def enable_qwen_rmsnorm_triton_patch():
    patched = 0
    try:
        from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm
    except Exception:
        Qwen3RMSNorm = None
    if Qwen3RMSNorm is not None and _patch_rmsnorm_class(Qwen3RMSNorm):
        patched += 1

    try:
        from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm
    except Exception:
        Qwen2RMSNorm = None
    if Qwen2RMSNorm is not None and _patch_rmsnorm_class(Qwen2RMSNorm):
        patched += 1

    return patched
