import logging
import os

import torch
import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

@triton.jit
def _add_tensor_scalar_kernel(
    x_ptr,
    out_ptr,
    scalar,
    alpha,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = x + scalar * alpha
    tl.store(out_ptr + offsets, y, mask=mask)


@triton.jit
def _add_broadcast_lastdim_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    alpha,
    last_dim,
    tiles_per_row,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    row = pid // tiles_per_row
    tile = pid - row * tiles_per_row
    row_start = row * last_dim
    offsets = row_start + tile * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < (row_start + last_dim)
    y = tl.load(y_ptr + row)
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    out = x + y * alpha
    tl.store(out_ptr + offsets, out, mask=mask)


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
def _arm_add_func(x, y, alpha):
    return x + y * alpha


@pointwise_dynamic(is_tensor=[True, False, False], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def _arm_add_func_tensor_scalar(x, y, alpha):
    return x + y * alpha


@pointwise_dynamic(is_tensor=[False, True, False], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def _arm_add_func_scalar_tensor(x, y, alpha):
    return x + y * alpha


def _base_add(A, B, *, alpha=1):
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
        block_size = 256 if A.dtype in (torch.float16, torch.bfloat16) else 128
        block_size = min(block_size, triton.next_power_of_2(max(total, 1)))
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
        return _arm_add_func(A, B, alpha)
    if isinstance(A, torch.Tensor):
        return _arm_add_func_tensor_scalar(A, B, alpha)
    if isinstance(B, torch.Tensor):
        return _arm_add_func_scalar_tensor(A, B, alpha)
    return torch.tensor(A + B * alpha)


def _base_add_(A, B, *, alpha=1):
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
        block_size = 256 if A.dtype in (torch.float16, torch.bfloat16) else 128
        block_size = min(block_size, triton.next_power_of_2(max(total, 1)))
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
        return _arm_add_func(A, B, alpha, out0=A)
    if isinstance(A, torch.Tensor):
        return _arm_add_func_tensor_scalar(A, B, alpha, out0=A)
    raise ValueError("Unreachable.")

def _use_triton(n_elements):
    return n_elements >= 4096


def _use_triton_scalar(n_elements):
    return n_elements >= 1


def _maybe_contiguous(x, y, out):
    if x.is_contiguous() and (y is None or y.is_contiguous()):
        return x, y, out, False
    x = x if x.is_contiguous() else x.contiguous()
    y = None if y is None else (y if y.is_contiguous() else y.contiguous())
    if out is not None and not out.is_contiguous():
        out = None
    return x, y, out, True


def _add_tensor_scalar_triton(x, scalar, alpha, out=None):
    n_elements = x.numel()
    if n_elements == 0:
        return x if out is None else out
    if n_elements == 1 and x.dtype is torch.bfloat16:
        val = float(x.item()) + float(scalar) * float(alpha)
        if out is None:
            out = torch.empty_like(x)
        out.fill_(val)
        return out
    block_size = 256 if x.dtype in (torch.float16, torch.bfloat16) else 128
    block_size = min(block_size, triton.next_power_of_2(max(n_elements, 1)))
    grid = (triton.cdiv(n_elements, block_size),)
    x_contig = x if x.is_contiguous() else x.contiguous()
    out_contig = out
    if out_contig is None:
        out_contig = torch.empty_like(x_contig)
    _add_tensor_scalar_kernel[grid](
        x_contig,
        out_contig,
        scalar,
        alpha,
        n_elements,
        BLOCK_SIZE=block_size,
        num_warps=1,
    )
    return out_contig


def _add_tensor_tensor_triton(x, y, alpha, out=None):
    n_elements = x.numel()
    if n_elements == 0:
        return x if out is None else out
    if n_elements == 1 and x.dtype is torch.bfloat16:
        val = float(x.item()) + float(y.item()) * float(alpha)
        if out is None:
            out = torch.empty_like(x)
        out.fill_(val)
        return out
    x_contig, y_contig, out_contig, _ = _maybe_contiguous(x, y, out)
    if out_contig is None:
        out_contig = torch.empty_like(x_contig)
    last_dim = x_contig.shape[-1]
    if (
        x_contig.ndim == 3
        and y_contig.ndim == 3
        and x_contig.shape[-1] > 1
        and y_contig.shape[-1] == 1
        and x_contig.shape[0] == y_contig.shape[0]
        and x_contig.shape[1] == y_contig.shape[1]
        and x_contig.is_contiguous()
        and y_contig.is_contiguous()
    ):
        block_size = 256 if x_contig.dtype in (torch.float16, torch.bfloat16) else 128
        block_size = min(block_size, triton.next_power_of_2(max(last_dim, 1)))
        tiles_per_row = triton.cdiv(last_dim, block_size)
        grid = (x_contig.shape[0] * x_contig.shape[1] * tiles_per_row,)
        _add_broadcast_lastdim_kernel[grid](
            x_contig,
            y_contig,
            out_contig,
            alpha,
            last_dim,
            tiles_per_row,
            BLOCK_SIZE=block_size,
            num_warps=1,
        )
        return out_contig
    if out is None:
        return _base_add(x, y, alpha=alpha)
    out.copy_(_base_add(x, y, alpha=alpha))
    return out


def _maybe_get_scalar_tensor(val):
    if isinstance(val, torch.Tensor) and val.numel() == 1:
        return val.item()
    return None


def _is_broadcast_lastdim(x, y):
    return (
        x.ndim == 3
        and y.ndim == 3
        and x.shape[0] == y.shape[0]
        and x.shape[1] == y.shape[1]
        and y.shape[2] == 1
        and x.shape[2] > 1
    )


def _is_lastdim1024_3d(x, y):
    return (
        x.ndim == 3
        and y.ndim == 3
        and x.shape == y.shape
        and x.shape[1] == 1
        and x.shape[2] == 1024
    )


def _add_tensor_tensor_3d_lastdim1024(x, y, alpha, out=None):
    n_elements = x.numel()
    if n_elements == 0:
        return x if out is None else out
    x_contig, y_contig, out_contig, _ = _maybe_contiguous(x, y, out)
    if out_contig is None:
        out_contig = torch.empty_like(x_contig)
    block_size = 256 if x_contig.dtype in (torch.float16, torch.bfloat16) else 128
    block_size = min(block_size, triton.next_power_of_2(max(n_elements, 1)))
    grid = (triton.cdiv(n_elements, block_size),)
    _add_contiguous_kernel[grid](
        x_contig,
        y_contig,
        out_contig,
        alpha,
        n_elements,
        BLOCK_SIZE=block_size,
        num_warps=1,
        num_stages=1,
    )
    return out_contig


def add(A, B, *, alpha=1):
    logging.debug("GEMS_ARM ADD")
    if isinstance(A, torch.Tensor) and not isinstance(B, torch.Tensor):
        if not _use_triton_scalar(A.numel()):
            return _base_add(A, B, alpha=alpha)
        return _add_tensor_scalar_triton(A, B, alpha)
    if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        scalar = _maybe_get_scalar_tensor(B)
        if scalar is not None:
            if not _use_triton_scalar(A.numel()):
                return _base_add(A, scalar, alpha=alpha)
            return _add_tensor_scalar_triton(A, scalar, alpha)
        if _use_triton(A.numel()) and _is_broadcast_lastdim(A, B):
            return _add_tensor_tensor_triton(A, B, alpha)
        if _use_triton(A.numel()) and _is_lastdim1024_3d(A, B):
            return _add_tensor_tensor_3d_lastdim1024(A, B, alpha)
        if alpha == 1 and _use_triton(B.numel()) and _is_broadcast_lastdim(B, A):
            return _add_tensor_tensor_triton(B, A, alpha)
        return _base_add(A, B, alpha=alpha)
    return _base_add(A, B, alpha=alpha)


def add_(A, B, *, alpha=1):
    logging.debug("GEMS_ARM ADD_")
    if isinstance(A, torch.Tensor) and not isinstance(B, torch.Tensor):
        if not _use_triton_scalar(A.numel()):
            return _base_add_(A, B, alpha=alpha)
        if A.is_contiguous():
            return _add_tensor_scalar_triton(A, B, alpha, out=A)
    if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        scalar = _maybe_get_scalar_tensor(B)
        if scalar is not None and A.is_contiguous():
            if not _use_triton_scalar(A.numel()):
                return _base_add_(A, scalar, alpha=alpha)
            return _add_tensor_scalar_triton(A, scalar, alpha, out=A)
        if _use_triton(A.numel()) and _is_broadcast_lastdim(A, B):
            return _add_tensor_tensor_triton(A, B, alpha, out=A)
        if _use_triton(A.numel()) and _is_lastdim1024_3d(A, B):
            return _add_tensor_tensor_3d_lastdim1024(A, B, alpha, out=A)
        if alpha == 1 and _use_triton(B.numel()) and _is_broadcast_lastdim(B, A):
            return _add_tensor_tensor_triton(B, A, alpha, out=A)
        return _base_add_(A, B, alpha=alpha)
    return _base_add_(A, B, alpha=alpha)
