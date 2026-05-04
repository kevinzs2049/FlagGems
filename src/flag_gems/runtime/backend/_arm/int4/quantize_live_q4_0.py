"""Live (in-memory) Q4_0-style W4 quantization (per-block-32, fp16 scale per
block per output channel) replacing each aligned nn.Linear with a
TLEInt4Q40Linear.

Mirrors llama.cpp's Q4_0 layout, which handles outlier-heavy weight rows
much better than per-channel W4 (the per-channel path was found to produce
unusable INT4 generation on Qwen3.5-4B due to a max/p99 outlier ratio
up to 7×).

Example:
    from flag_gems.runtime.backend._arm.int4 import quantize_q4_0_and_replace_linears
    quantize_q4_0_and_replace_linears(m)
"""
import logging
from typing import Iterable, Optional

import torch

from .tle_int4_q40_linear import TLEInt4Q40Linear, quantize_w4_q4_0_per_block

logger = logging.getLogger(__name__)


def quantize_q4_0_and_replace_linears(
    model: torch.nn.Module,
    skip: Optional[Iterable[str]] = None,
    require_K_divisible_by: int = 32,
    require_N_divisible_by: int = 4,
    skip_with_bias: bool = True,
) -> int:
    """Walk the model, in-place Q4_0-quantize each aligned nn.Linear weight,
    and swap it for a TLEInt4Q40Linear.

    Args:
        model: any torch.nn.Module (typically a transformers model).
        skip: module names to leave alone.
        require_K_divisible_by: K must be a multiple of 32 (block size).
        require_N_divisible_by: N must be a multiple of 4 (SDOT lane).
        skip_with_bias: TLEInt4Q40Linear has no bias parameter.

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
            raise ValueError(f"{name} has bias=True; TLEInt4Q40Linear has no bias")

        N, K = module.weight.shape
        if K % require_K_divisible_by != 0 or N % require_N_divisible_by != 0:
            n_skipped_align += 1
            logger.debug(
                "quantize_q4_0_and_replace_linears: %s K=%d N=%d not aligned "
                "(K%%%d=%d, N%%%d=%d)",
                name, K, N, require_K_divisible_by, K % require_K_divisible_by,
                require_N_divisible_by, N % require_N_divisible_by,
            )
            continue

        # nn.Linear stores weight as [N, K]; quantize_w4_q4_0_per_block expects
        # [K, N], so transpose first.
        w_kn = module.weight.data.t().contiguous()  # [K, N]
        w_int4_kn, block_scales_fp16 = quantize_w4_q4_0_per_block(w_kn)

        parts = name.split(".")
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        setattr(parent, parts[-1], TLEInt4Q40Linear(w_int4_kn, block_scales_fp16))
        del module
        n_replaced += 1

    logger.info(
        "quantize_q4_0_and_replace_linears: replaced %d Linears "
        "(skipped: %d alignment, %d bias, %d explicit)",
        n_replaced, n_skipped_align, n_skipped_bias, len(skip_set),
    )
    return n_replaced
