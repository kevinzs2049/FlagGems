import logging
import os

import torch
import triton
import triton.language as tl

from flag_gems.ops.add import add as base_add
from flag_gems.ops.add import add_ as base_add_

CPU_BLOCK_SIZE = 4096
CPU_ST_THRESHOLD = 65536
TILE_SIZE = 16

_ADD_CPU_CONFIGS = [
    triton.Config({"TILE_SIZE": 16, "BLOCK_SIZE": 128}, num_threads=1),
    triton.Config({"TILE_SIZE": 16, "BLOCK_SIZE": 128}, num_threads=0),
]


@triton.autotune(configs=_ADD_CPU_CONFIGS, key=["n_elements"])
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
def _add_row_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    alpha,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
): 
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    out = x + y * alpha
    tl.store(out_ptr + offsets, out, mask=mask)


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


def _use_triton(n_elements):
    if os.environ.get("GEMS_ARM_ADD_FORCE_TRITON") == "1":
        return True
    return n_elements >= 4096


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
        last_dim in (128, 1024)
        and n_elements % last_dim == 0
        and x_contig.is_contiguous()
        and y_contig.is_contiguous()
    ):
        grid = (n_elements // last_dim,)
        _add_row_kernel[grid](
            x_contig,
            y_contig,
            out_contig,
            alpha,
            n_elements,
            BLOCK_SIZE=last_dim,
            num_warps=1,
        )
        return out_contig
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
        if os.environ.get("GEMS_DEBUG_ADD_BCAST") == "1":
            print(
                "[GEMS_DEBUG_ADD_BCAST] "
                f"x={tuple(x_contig.shape)} y={tuple(y_contig.shape)}"
            )
        block_size = 1024 if x_contig.dtype in (torch.float16, torch.bfloat16) else 512
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
        return base_add(x, y, alpha=alpha)
    out.copy_(base_add(x, y, alpha=alpha))
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


def add(A, B, *, alpha=1):
    logging.debug("GEMS_ARM ADD")
    if os.environ.get("GEMS_DEBUG_ADD") == "1":
        a_shape = tuple(A.shape) if isinstance(A, torch.Tensor) else None
        b_shape = tuple(B.shape) if isinstance(B, torch.Tensor) else None
        print(f"[GEMS_DEBUG_ADD] add: A={a_shape} B={b_shape} alpha={alpha}")
    if os.environ.get("GEMS_ARM_ADD_TRITON") != "1":
        if os.environ.get("GEMS_DEBUG_ADD") == "1":
            print("[GEMS_DEBUG_ADD] GEMS_ARM_ADD_TRITON!=1, fallback to base_add")
        return base_add(A, B, alpha=alpha)
    if isinstance(A, torch.Tensor) and not isinstance(B, torch.Tensor):
        if not _use_triton(A.numel()):
            return base_add(A, B, alpha=alpha)
        return _add_tensor_scalar_triton(A, B, alpha)
    if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        scalar = _maybe_get_scalar_tensor(B)
        if scalar is not None:
            if not _use_triton(A.numel()):
                return base_add(A, scalar, alpha=alpha)
            return _add_tensor_scalar_triton(A, scalar, alpha)
        if _use_triton(A.numel()) and _is_broadcast_lastdim(A, B):
            return _add_tensor_tensor_triton(A, B, alpha)
        if alpha == 1 and _use_triton(B.numel()) and _is_broadcast_lastdim(B, A):
            return _add_tensor_tensor_triton(B, A, alpha)
        if not _use_triton(A.numel()):
            return base_add(A, B, alpha=alpha)
        if (
            A.shape == B.shape
            and A.dtype == B.dtype
            and A.device == B.device
        ):
            return _add_tensor_tensor_triton(A, B, alpha)
    return base_add(A, B, alpha=alpha)


def add_(A, B, *, alpha=1):
    logging.debug("GEMS_ARM ADD_")
    if os.environ.get("GEMS_ARM_ADD_TRITON") != "1":
        return base_add_(A, B, alpha=alpha)
    if isinstance(A, torch.Tensor) and not isinstance(B, torch.Tensor):
        if not _use_triton(A.numel()):
            return base_add_(A, B, alpha=alpha)
        if A.is_contiguous():
            return _add_tensor_scalar_triton(A, B, alpha, out=A)
    if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        scalar = _maybe_get_scalar_tensor(B)
        if scalar is not None and A.is_contiguous():
            if not _use_triton(A.numel()):
                return base_add_(A, scalar, alpha=alpha)
            return _add_tensor_scalar_triton(A, scalar, alpha, out=A)
        if _use_triton(A.numel()) and _is_broadcast_lastdim(A, B):
            return _add_tensor_tensor_triton(A, B, alpha, out=A)
        if alpha == 1 and _use_triton(B.numel()) and _is_broadcast_lastdim(B, A):
            return _add_tensor_tensor_triton(B, A, alpha, out=A)
        if not _use_triton(A.numel()):
            return base_add_(A, B, alpha=alpha)
        if (
            A.is_contiguous()
            and B.is_contiguous()
            and A.shape == B.shape
            and A.dtype == B.dtype
            and A.device == B.device
        ):
            return _add_tensor_tensor_triton(A, B, alpha, out=A)
    return base_add_(A, B, alpha=alpha)
