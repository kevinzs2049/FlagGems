"""Live (in-memory) W4 per-channel symmetric quantization of nn.Linear,
replacing each with a TLEInt4Linear.

Per-output-channel symmetric: scale = max(|w|) / 7 per row, weights clamped
to [-7, 7]. (4-bit signed range is [-8, 7], but we use [-7, 7] symmetric to
match common practice — costs ~1 LSB of dynamic range, gains numeric
stability across pack/unpack round-trip.)

Activation quant is dynamic per-token int8, handled inside TLEInt4Linear.

Example:
    from transformers import AutoModelForCausalLM
    from flag_gems.runtime.backend._arm.int4 import quantize_w4_and_replace_linears
    m = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-2B", dtype=torch.bfloat16)
    quantize_w4_and_replace_linears(m)
"""
import logging
from typing import Iterable, Optional, Tuple

import torch

from .tle_int4_linear import TLEInt4Linear

logger = logging.getLogger(__name__)


def _quantize_weight_w4_per_channel_sym(w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-output-channel symmetric W4 quant.

    Returns:
        w_int4:  [N, K] int8 with values in [-7, 7]
        w_scale: [N]    fp32
    """
    w_fp32 = w.detach().to(torch.float32)
    absmax = w_fp32.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)  # [N, 1]
    scale = absmax / 7.0
    w_int4 = (w_fp32 / scale).round().clamp(-7, 7).to(torch.int8)
    return w_int4, scale.squeeze(-1).contiguous().to(torch.float32)


def quantize_w4_and_replace_linears(
    model: torch.nn.Module,
    skip: Optional[Iterable[str]] = None,
    require_divisible_by: int = 4,
    skip_with_bias: bool = True,
) -> int:
    """In-memory W4 quantize each aligned nn.Linear and swap it for TLEInt4Linear.

    Args:
        model: any torch.nn.Module
        skip: module names to leave alone
        require_divisible_by: SDOT requires K%4==N%4==0
        skip_with_bias: TLEInt4Linear has no bias parameter

    Returns: number of Linears replaced.
    """
    skip_set = set(skip) if skip else set()
    n_replaced = 0
    n_skipped_align = 0
    n_skipped_bias = 0

    for name, module in list(model.named_modules()):
        if not isinstance(module, torch.nn.Linear):
            continue
        if name in skip_set:
            continue
        if module.bias is not None:
            if skip_with_bias:
                n_skipped_bias += 1
                continue
            raise ValueError(f"{name} has bias=True; TLEInt4Linear has no bias")

        N, K = module.weight.shape
        if K % require_divisible_by != 0 or N % require_divisible_by != 0:
            n_skipped_align += 1
            continue

        w_int4, w_scale = _quantize_weight_w4_per_channel_sym(module.weight.data)

        parts = name.split(".")
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        setattr(parent, parts[-1], TLEInt4Linear(w_int4, w_scale))
        del module
        n_replaced += 1

    logger.info(
        "quantize_w4_and_replace_linears: replaced %d Linears "
        "(skipped: %d alignment, %d bias, %d explicit)",
        n_replaced, n_skipped_align, n_skipped_bias, len(skip_set),
    )
    return n_replaced
