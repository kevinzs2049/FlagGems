# import itertools
import logging
from typing import List, Tuple, Union

import torch
import triton
import triton.language as tl

SMALL_CAT_NUMEL_THRESHOLD = 4_096

@triton.jit
def cat_kernel(
    in_ptr,
    out_ptr,
    in_strides_ptr,
    out_strides_ptr,
    shape_ptr,
    offset,
    total_elements,
    ndim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < total_elements

    linear_id = offs
    in_offset = tl.zeros([BLOCK_SIZE], dtype=tl.int32)
    out_offset = tl.full([BLOCK_SIZE], offset, dtype=tl.int32)

    for d in range(ndim):
        shape_d = tl.load(shape_ptr + d)
        in_stride = tl.load(in_strides_ptr + d)
        out_stride = tl.load(out_strides_ptr + d)

        coord = linear_id % shape_d
        linear_id = linear_id // shape_d

        in_offset += coord * in_stride
        out_offset += coord * out_stride

    val = tl.load(in_ptr + in_offset, mask=mask)
    tl.store(out_ptr + out_offset, val, mask=mask)

@triton.jit
def cat_kernel_dim0_contig(
        in_ptr,
        out_ptr,
        offset,
        total_elements,
        BLOCK_SIZE: tl.constexpr,
):
    logging.debug("GEMS CAT cat_kernel_dim0_contig")
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < total_elements
    tl.store(out_ptr + offset + offs, tl.load(in_ptr + offs, mask=mask), mask=mask)


def cat(
    A: Union[Tuple[torch.Tensor, ...], List[torch.Tensor]], dim: int = 0
) -> torch.Tensor:
    logging.debug("TRITON CAT")

    if len(A) == 0:
        raise RuntimeError("torch.cat(): expected a non-empty list of Tensors")
    if len(A) == 1:
        return A[0]

    dim = dim % A[0].ndim
    base_shape = A[0].shape

    contiguous_dim0 = (
            dim == 0
            and all(
        t.is_contiguous() and t.device == A[0].device and t.dtype == A[0].dtype
        for t in A
    )
    )


    for t in A:
        if t.ndim != len(base_shape):
            raise RuntimeError("Tensors must have the same number of dimensions")
        for i in range(t.ndim):
            if i != dim and t.shape[i] != base_shape[i]:
                raise RuntimeError(f"Size mismatch at dim {i}")

    if all(t.numel() == 0 for t in A):
        empty_shape = list(base_shape)
        empty_shape[dim] = 0
        return torch.empty(empty_shape, dtype=A[0].dtype, device=A[0].device)

    total_numel = sum(t.numel() for t in A)
    out_shape = list(base_shape)
    out_shape[dim] = sum(t.shape[dim] for t in A)
    out = torch.empty(out_shape, dtype=A[0].dtype, device=A[0].device)

    if all(t.is_contiguous() and t.device == out.device and t.dtype == out.dtype for t in A):
        offset_dim = 0
        for t in A:
            if t.numel() == 0:
                continue
            out_slice = out.narrow(dim, offset_dim, t.shape[dim])
            out_slice.copy_(t)
            offset_dim += t.shape[dim]
        return out

    if total_numel <= SMALL_CAT_NUMEL_THRESHOLD:
        offset_dim = 0
        for t in A:
            if t.numel() == 0:
                continue
            out_slice = out.narrow(dim, offset_dim, t.shape[dim])
            out_slice.copy_(t)
            offset_dim += t.shape[dim]
        return out

    out_strides = torch.tensor(out.stride(), dtype=torch.int32, device=out.device)
    offset = 0

    for t in A:
        if t.numel() == 0:
            continue

        if contiguous_dim0:
            # Specialized contiguous dim-0 copy avoids per-dimension stride math
            # and is cheaper to launch on CPU Triton backend.
            total_elements = t.numel()
            block_size = 256 if t.device.type == "cpu" else 128
            grid = lambda META: (triton.cdiv(total_elements, META["BLOCK_SIZE"]),)
            cat_kernel_dim0_contig[grid](
                in_ptr=t,
                out_ptr=out,
                offset=offset,
                total_elements=total_elements,
                BLOCK_SIZE=block_size,
            )
            offset += t.shape[0] * out.stride()[0]
            continue

        in_strides = torch.tensor(t.stride(), dtype=torch.int32, device=t.device)
        shape_tensor = torch.tensor(t.shape, dtype=torch.int32, device=t.device)
        total_elements = t.numel()

        # Use a larger tile on CPU to better amortize kernel launch overhead and
        # enable vectorization. Keep a smaller tile elsewhere to avoid bloating
        # register pressure on GPU backends.
        block_size = 16 if t.device.type == "cpu" else 128
        grid = lambda META: (triton.cdiv(total_elements, META["BLOCK_SIZE"]),)
        cat_kernel[grid](
            in_ptr=t,
            out_ptr=out,
            in_strides_ptr=in_strides,
            out_strides_ptr=out_strides,
            shape_ptr=shape_tensor,
            offset=offset,
            total_elements=total_elements,
            ndim=t.ndim,
            BLOCK_SIZE=block_size,
        )

        offset += t.shape[dim] * out.stride()[dim]

    return out
