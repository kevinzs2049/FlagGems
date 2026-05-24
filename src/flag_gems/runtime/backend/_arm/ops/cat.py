import logging
from typing import List, Tuple, Union

import torch


def cat(
    A: Union[Tuple[torch.Tensor, ...], List[torch.Tensor]], dim: int = 0
) -> torch.Tensor:
    logging.debug("GEMS CAT")

    if len(A) == 0:
        raise RuntimeError("torch.cat(): expected a non-empty list of Tensors")
    if len(A) == 1:
        return A[0]

    # Match PyTorch behavior: allow 1D empty tensors to participate in cat.
    normal_tensors = [t for t in A if not (t.ndim == 1 and t.numel() == 0)]
    if len(normal_tensors) == 0:
        return torch.empty((0,), dtype=A[0].dtype, device=A[0].device)

    n = len(normal_tensors)
    t0 = normal_tensors[0]
    ndim = t0.ndim
    dim = dim % ndim
    base_shape = t0.shape
    dtype = t0.dtype
    device = t0.device

    for t in normal_tensors:
        if t.ndim != ndim:
            raise RuntimeError("Tensors must have the same number of dimensions")
        for i in range(ndim):
            if i != dim and t.shape[i] != base_shape[i]:
                raise RuntimeError(f"Size mismatch at dim {i}")

    if all(t.numel() == 0 for t in normal_tensors):
        empty_shape = list(base_shape)
        empty_shape[dim] = 0
        return torch.empty(empty_shape, dtype=dtype, device=device)

    out_shape = list(base_shape)
    out_shape[dim] = sum(t.shape[dim] for t in normal_tensors)
    out = torch.empty(out_shape, dtype=dtype, device=device)
    out_strides = out.stride()
    dim_stride = out_strides[dim]

    # Use as_strided + copy_ instead of narrow + copy_.
    #
    # Replacing the original Triton kernel (ndim UDIV per element ≈ 80-160
    # cycles/element for 4D tensors, plus ~9μs launch overhead) with Python-level
    # copy using as_strided views.  as_strided is ~0.37μs cheaper than narrow per
    # call and avoids the Triton kernel dispatch entirely.
    #
    # KV-cache case ([1,H,S,D]+[1,H,1,D], dim=2, contiguous):
    #   old Triton:   120-600 μs  →  this impl: ~10 μs (vs ATen ~2 μs)
    #
    # Remaining gap vs ATen: ATen executes malloc+memcpy fully in C++ with no
    # Python round-trips; from Python we have ~3 ops × ~2μs dispatch = 6μs floor.
    # For medium/large tensors (S≥100) the copy time dominates and both converge.
    offset = 0
    for t in normal_tensors:
        if t.numel() == 0:
            continue
        size = t.shape[dim]
        # as_strided view into the output slice for this tensor
        torch.as_strided(out, t.shape, out_strides, offset * dim_stride).copy_(t)
        offset += size

    return out
