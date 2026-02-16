import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@triton.jit
def _mul_contiguous_kernel(x_ptr, y_ptr, out_ptr, total_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < total_elements
    x = tl.load(x_ptr + idx, mask=mask, other=0)
    y = tl.load(y_ptr + idx, mask=mask, other=0)
    tl.store(out_ptr + idx, x * y, mask=mask)


@triton.jit
def _mul_contiguous_single_program_kernel(
    x_ptr, y_ptr, out_ptr, total_elements, BLOCK_SIZE: tl.constexpr
):
    offs = tl.arange(0, BLOCK_SIZE)
    for base in range(0, total_elements, BLOCK_SIZE):
        idx = base + offs
        mask = idx < total_elements
        x = tl.load(x_ptr + idx, mask=mask, other=0)
        y = tl.load(y_ptr + idx, mask=mask, other=0)
        tl.store(out_ptr + idx, x * y, mask=mask)


@triton.jit(do_not_specialize=["scalar"])
def _mul_tensor_scalar_kernel(
    x_ptr, out_ptr, scalar, total_elements, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < total_elements
    x = tl.load(x_ptr + idx, mask=mask, other=0)
    tl.store(out_ptr + idx, x * scalar, mask=mask)


@triton.jit(do_not_specialize=["scalar"])
def _mul_tensor_scalar_single_program_kernel(
    x_ptr, out_ptr, scalar, total_elements, BLOCK_SIZE: tl.constexpr
):
    offs = tl.arange(0, BLOCK_SIZE)
    for base in range(0, total_elements, BLOCK_SIZE):
        idx = base + offs
        mask = idx < total_elements
        x = tl.load(x_ptr + idx, mask=mask, other=0)
        tl.store(out_ptr + idx, x * scalar, mask=mask)


@triton.jit
def _mul_lastdim_vector_kernel(
    x_ptr, y_ptr, out_ptr, total_elements, last_dim, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < total_elements
    col = idx % last_dim
    x = tl.load(x_ptr + idx, mask=mask, other=0)
    y = tl.load(y_ptr + col, mask=mask, other=0)
    tl.store(out_ptr + idx, x * y, mask=mask)


@triton.jit
def _mul_lastdim_vector_single_program_kernel(
    x_ptr, y_ptr, out_ptr, total_elements, last_dim, BLOCK_SIZE: tl.constexpr
):
    offs = tl.arange(0, BLOCK_SIZE)
    for base in range(0, total_elements, BLOCK_SIZE):
        idx = base + offs
        mask = idx < total_elements
        col = idx % last_dim
        x = tl.load(x_ptr + idx, mask=mask, other=0)
        y = tl.load(y_ptr + col, mask=mask, other=0)
        tl.store(out_ptr + idx, x * y, mask=mask)


@triton.jit
def _mul_lastdim_rowscalar_kernel(
    x_ptr, y_ptr, out_ptr, total_elements, last_dim, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < total_elements
    row = idx // last_dim
    x = tl.load(x_ptr + idx, mask=mask, other=0)
    y = tl.load(y_ptr + row, mask=mask, other=0)
    tl.store(out_ptr + idx, x * y, mask=mask)


@triton.jit
def _mul_lastdim_rowscalar_single_program_kernel(
    x_ptr, y_ptr, out_ptr, total_elements, last_dim, BLOCK_SIZE: tl.constexpr
):
    offs = tl.arange(0, BLOCK_SIZE)
    for base in range(0, total_elements, BLOCK_SIZE):
        idx = base + offs
        mask = idx < total_elements
        row = idx // last_dim
        x = tl.load(x_ptr + idx, mask=mask, other=0)
        y = tl.load(y_ptr + row, mask=mask, other=0)
        tl.store(out_ptr + idx, x * y, mask=mask)


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def _mul_tensor_tensor_fallback(x, y):
    return x * y


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def _mul_tensor_scalar_fallback(x, y):
    return x * y


def _supported_fast_dtype(dtype: torch.dtype) -> bool:
    return dtype in (torch.bfloat16, torch.float32, torch.float64)


def _single_program_block(n_elements: int) -> int:
    if n_elements <= 256:
        return 32
    if n_elements <= 2048:
        return 128
    return 256


def _use_single_program(n_elements: int) -> bool:
    # Decode path tensors are small; a single-program loop avoids launch overhead.
    return 1 < n_elements <= 262144


def _launch_contiguous_mul(x: torch.Tensor, y: torch.Tensor, out: torch.Tensor) -> None:
    total = x.numel()
    if _use_single_program(total):
        block = _single_program_block(total)
        _mul_contiguous_single_program_kernel[(1,)](
            x, y, out, total, BLOCK_SIZE=block, num_warps=1, num_stages=1
        )
        return
    grid = lambda META: (triton.cdiv(total, META["BLOCK_SIZE"]),)
    _mul_contiguous_kernel[grid](
        x, y, out, total, BLOCK_SIZE=256, num_warps=1, num_stages=1
    )


def _launch_tensor_scalar_mul(x: torch.Tensor, scalar: float, out: torch.Tensor) -> None:
    total = x.numel()
    if _use_single_program(total):
        block = _single_program_block(total)
        _mul_tensor_scalar_single_program_kernel[(1,)](
            x, out, scalar, total, BLOCK_SIZE=block, num_warps=1, num_stages=1
        )
        return
    grid = lambda META: (triton.cdiv(total, META["BLOCK_SIZE"]),)
    _mul_tensor_scalar_kernel[grid](
        x, out, scalar, total, BLOCK_SIZE=256, num_warps=1, num_stages=1
    )


def _launch_lastdim_vector_mul(
    x: torch.Tensor, y: torch.Tensor, out: torch.Tensor, last_dim: int
) -> None:
    total = x.numel()
    if _use_single_program(total):
        block = _single_program_block(total)
        _mul_lastdim_vector_single_program_kernel[(1,)](
            x,
            y,
            out,
            total,
            last_dim,
            BLOCK_SIZE=block,
            num_warps=1,
            num_stages=1,
        )
        return
    grid = lambda META: (triton.cdiv(total, META["BLOCK_SIZE"]),)
    _mul_lastdim_vector_kernel[grid](
        x, y, out, total, last_dim, BLOCK_SIZE=256, num_warps=1, num_stages=1
    )


def _launch_lastdim_rowscalar_mul(
    x: torch.Tensor, y_rows: torch.Tensor, out: torch.Tensor, last_dim: int
) -> None:
    total = x.numel()
    if _use_single_program(total):
        block = _single_program_block(total)
        _mul_lastdim_rowscalar_single_program_kernel[(1,)](
            x,
            y_rows,
            out,
            total,
            last_dim,
            BLOCK_SIZE=block,
            num_warps=1,
            num_stages=1,
        )
        return
    grid = lambda META: (triton.cdiv(total, META["BLOCK_SIZE"]),)
    _mul_lastdim_rowscalar_kernel[grid](
        x, y_rows, out, total, last_dim, BLOCK_SIZE=256, num_warps=1, num_stages=1
    )


def _try_mul_fastpath(lhs: torch.Tensor, rhs: torch.Tensor, out: torch.Tensor) -> bool:
    if lhs.device.type != "cpu" or rhs.device != lhs.device:
        return False
    if lhs.dtype != rhs.dtype or not _supported_fast_dtype(lhs.dtype):
        return False
    if not lhs.is_contiguous() or not out.is_contiguous():
        return False
    if lhs.numel() == 0:
        return False
    if lhs.shape != out.shape:
        return False
    if torch._C._is_alias_of(lhs, rhs) and lhs.shape != rhs.shape:
        return False

    if rhs.is_contiguous() and rhs.shape == lhs.shape:
        _launch_contiguous_mul(lhs, rhs, out)
        return True

    if rhs.numel() == 1 and rhs.is_contiguous():
        _launch_tensor_scalar_mul(lhs, float(rhs.item()), out)
        return True

    if lhs.ndim == 0:
        return False

    last_dim = int(lhs.shape[-1])
    if last_dim <= 0:
        return False

    if rhs.is_contiguous() and rhs.ndim == 1 and int(rhs.shape[0]) == last_dim:
        _launch_lastdim_vector_mul(lhs, rhs, out, last_dim)
        return True

    if (
        rhs.is_contiguous()
        and rhs.ndim == lhs.ndim
        and rhs.shape[:-1] == lhs.shape[:-1]
        and int(rhs.shape[-1]) == 1
    ):
        _launch_lastdim_rowscalar_mul(lhs, rhs.view(-1), out, last_dim)
        return True

    return False


def mul(A, B):
    logger.debug("GEMS MUL")

    if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        if A.numel() >= B.numel():
            out = torch.empty_like(A) if A.shape == torch.broadcast_shapes(A.shape, B.shape) else None
            if out is not None and _try_mul_fastpath(A, B, out):
                return out
        if B.numel() > A.numel():
            out = torch.empty_like(B) if B.shape == torch.broadcast_shapes(A.shape, B.shape) else None
            if out is not None and _try_mul_fastpath(B, A, out):
                return out
        return _mul_tensor_tensor_fallback(A, B)
    if isinstance(A, torch.Tensor):
        return _mul_tensor_scalar_fallback(A, B)
    if isinstance(B, torch.Tensor):
        return _mul_tensor_scalar_fallback(B, A)
    return torch.tensor(A * B)


def mul_(A, B):
    logger.debug("GEMS MUL_")

    if isinstance(B, torch.Tensor):
        if _try_mul_fastpath(A, B, A):
            return A
        return _mul_tensor_tensor_fallback(A, B, out0=A)
    return _mul_tensor_scalar_fallback(A, B, out0=A)
