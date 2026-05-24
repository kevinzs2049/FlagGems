import os
from contextlib import contextmanager

import torch
import torch.nn.functional as F


def _is_arm_cpu_runtime(device: str) -> bool:
    if device != "cpu":
        return False
    vendor_env = os.getenv("GEMS_VENDOR", "").strip().lower()
    if vendor_env:
        return vendor_env == "arm"
    try:
        import flag_gems

        vendor_name = str(getattr(flag_gems.runtime.device, "vendor_name", "")).lower()
        return vendor_name == "arm"
    except Exception:
        return False


def enable_linear_m1_fastpath_for_run(
    *,
    enabled: bool,
    scope: str,
    use_gems: bool,
    device: str = "cpu",
) -> bool:
    if not enabled:
        return False
    scope = scope.strip().lower()
    if scope == "gems":
        return use_gems and _is_arm_cpu_runtime(device)
    if scope == "both":
        return _is_arm_cpu_runtime(device)
    return False


@contextmanager
def linear_m1_fastpath(enabled: bool):
    stats = {"hits": 0}
    if not enabled:
        yield stats
        return

    orig_linear = F.linear

    def _linear_m1_dispatch(input, weight, bias=None):
        if (
            input.device.type != "cpu"
            or weight.device.type != "cpu"
            or input.ndim < 2
            or input.shape[-1] != weight.shape[-1]
            or input.shape[-2] != 1
        ):
            return orig_linear(input, weight, bias)

        # Decode path: one row matmul -> vector matmul.
        if input.numel() != input.shape[-1]:
            return orig_linear(input, weight, bias)

        out = torch.mv(weight, input.reshape(-1))
        if bias is not None:
            out = out + bias
        stats["hits"] += 1
        return out.reshape(*input.shape[:-1], weight.shape[0])

    F.linear = _linear_m1_dispatch
    try:
        yield stats
    finally:
        F.linear = orig_linear
