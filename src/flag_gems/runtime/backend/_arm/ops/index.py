import logging
import os

import torch

from flag_gems.ops.index import index as base_index

from .index_select import index_select

_INDEX_DEBUG_ENABLED = os.environ.get("GEMS_DEBUG_INDEX") == "1"


def index(inp, indices):
    logging.debug("GEMS_ARM INDEX")
    if _INDEX_DEBUG_ENABLED:
        summary = []
        for item in indices:
            if item is None:
                summary.append("None")
            elif isinstance(item, torch.Tensor):
                summary.append(f"Tensor(shape={tuple(item.shape)},dtype={item.dtype})")
            else:
                summary.append(type(item).__name__)
        print(f"[GEMS_DEBUG_INDEX] inp_shape={tuple(inp.shape)} indices={summary}")

    if (
        isinstance(inp, torch.Tensor)
        and inp.device.type == "cpu"
        and isinstance(indices, (list, tuple))
        and inp.ndim == 2
        and len(indices) == 2
        and indices[0] is None
    ):
        if (
            isinstance(indices[1], torch.Tensor)
            and indices[1].ndim == 1
            and indices[1].numel() == 1
            and indices[1].dtype in (torch.int64, torch.int32, torch.int16, torch.int8)
        ):
            col = int(indices[1].item())
            if col < 0 or col >= inp.shape[1]:
                return torch.zeros(
                    (inp.shape[0], 1), dtype=inp.dtype, device=inp.device
                )
            col_index = torch.tensor([col], dtype=torch.long, device=inp.device)
            return index_select(inp, 1, col_index)
    return base_index(inp, indices)
