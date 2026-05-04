"""Live Q4_0 v2 quantization (llama.cpp-style per-N K-major layout)."""
import logging
from typing import Iterable, Optional

import torch

from .tle_int4_q40_v2_linear import TLEInt4Q40V2Linear, quantize_w4_q4_0_v2

logger = logging.getLogger(__name__)


def quantize_q4_0_v2_and_replace_linears(
    model: torch.nn.Module,
    skip: Optional[Iterable[str]] = None,
    require_K_divisible_by: int = 32,
    require_N_divisible_by: int = 4,
    skip_with_bias: bool = True,
) -> int:
    """Walk the model, in-place Q4_0-v2-quantize each aligned nn.Linear and
    swap it for a TLEInt4Q40V2Linear (which carries the new [N, K/32, 18]
    packed layout)."""
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
            raise ValueError(f"{name} has bias=True; TLEInt4Q40V2Linear has no bias")

        N, K = module.weight.shape
        if K % require_K_divisible_by != 0 or N % require_N_divisible_by != 0:
            n_skipped_align += 1
            continue

        # Linear stores [N, K]; quantize_w4_q4_0_v2 expects [K, N]
        w_kn = module.weight.data.t().contiguous()
        w_int_unsigned, block_scales = quantize_w4_q4_0_v2(w_kn)

        parts = name.split(".")
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        setattr(parent, parts[-1], TLEInt4Q40V2Linear(w_int_unsigned, block_scales))
        del module
        n_replaced += 1

    logger.info(
        "quantize_q4_0_v2_and_replace_linears: replaced %d Linears "
        "(skipped %d alignment, %d bias, %d explicit)",
        n_replaced, n_skipped_align, n_skipped_bias, len(skip_set),
    )
    return n_replaced
