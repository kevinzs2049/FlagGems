import logging

import torch
import triton
import triton.language as tl

from ..utils import pointwise_dynamic

@triton.jit
def _add_contiguous_kernel(
        x_ptr,
        y_ptr,
        out_ptr,
        alpha,
        total_elements,
        BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < total_elements
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y * alpha, mask=mask)

@pointwise_dynamic(is_tensor=[True, True, False], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def add_func(x, y, alpha):
    return x + y * alpha


@pointwise_dynamic(
    is_tensor=[True, False, False], promotion_methods=[(0, 1, "DEFAULT")]
)
@triton.jit
def add_func_tensor_scalar(x, y, alpha):
    return x + y * alpha


@pointwise_dynamic(
    is_tensor=[False, True, False], promotion_methods=[(0, 1, "DEFAULT")]
)
@triton.jit
def add_func_scalar_tensor(x, y, alpha):
    return x + y * alpha


def add(A, B, *, alpha=1):
    logging.debug("GEMS ADD")
    if (
            isinstance(A, torch.Tensor)
            and isinstance(B, torch.Tensor)
            and A.device.type == "cpu"
            and B.device == A.device
            and A.is_contiguous()
            and B.is_contiguous()
            and A.shape == B.shape
            and A.dtype == B.dtype
            and A.dtype in (torch.bfloat16, torch.float32, torch.float64)
            and A.untyped_storage().data_ptr() != B.untyped_storage().data_ptr()
    ):
        total = A.numel()
        alpha_cast = float(alpha)
        block_size = 256
        grid = lambda META: (triton.cdiv(total, META["BLOCK_SIZE"]),)
        _add_contiguous_kernel[grid](
            A,
            B,
            A,
            alpha_cast,
            total,
            BLOCK_SIZE=block_size,
            num_warps=1,
            num_stages=1,
        )
        return A

    if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        return add_func(A, B, alpha)
    elif isinstance(A, torch.Tensor):
        return add_func_tensor_scalar(A, B, alpha)
    elif isinstance(B, torch.Tensor):
        return add_func_scalar_tensor(A, B, alpha)
    else:
        return torch.tensor(A + B * alpha)


def add_(A, B, *, alpha=1):
    logging.debug("GEMS ADD_")
    if (
            isinstance(A, torch.Tensor)
            and isinstance(B, torch.Tensor)
            and A.device.type == "cpu"
            and B.device == A.device
            and A.is_contiguous()
            and B.is_contiguous()
            and A.shape == B.shape
            and A.dtype == B.dtype
            and A.dtype in (torch.bfloat16, torch.float32, torch.float64)
            and A.untyped_storage().data_ptr() != B.untyped_storage().data_ptr()
    ):
        total = A.numel()
        alpha_cast = float(alpha)
        block_size = 256
        grid = lambda META: (triton.cdiv(total, META["BLOCK_SIZE"]),)
        _add_contiguous_kernel[grid](
            A,
            B,
            A,
            alpha_cast,
            total,
            BLOCK_SIZE=block_size,
            num_warps=1,
            num_stages=1,
        )
        return A
    if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        return add_func(A, B, alpha, out0=A)
    elif isinstance(A, torch.Tensor):
        return add_func_tensor_scalar(A, B, alpha, out0=A)
    # elif isinstance(B, torch.Tensor):
    #     return add_func_scalar_tensor(A, B, alpha, out0=A)
    else:
        raise ValueError("Unreachable.")
