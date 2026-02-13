import logging
import os

import torch
import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

_PREWARM_ADD_DONE = False
_ADD_PREWARM_ENABLED = os.environ.get("GEMS_ARM_ADD_PREWARM", "1") == "1"
_ADD_TRITON_ENABLED = os.environ.get("GEMS_ARM_ADD_TRITON", "1") == "1"


@triton.jit(do_not_specialize=["scalar", "alpha", "n_elements"])
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


@triton.jit(do_not_specialize=["scalar", "alpha", "n_elements"])
def _add_tensor_scalar_single_program_kernel(
    x_ptr,
    out_ptr,
    scalar,
    alpha,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    offs = tl.arange(0, BLOCK_SIZE)
    for base in range(0, n_elements, BLOCK_SIZE):
        idx = base + offs
        mask = idx < n_elements
        x = tl.load(x_ptr + idx, mask=mask, other=0.0)
        y = x + scalar * alpha
        tl.store(out_ptr + idx, y, mask=mask)


@triton.jit(do_not_specialize=["scalar", "alpha", "n_elements"])
def _add_tensor_scalar_tiny_kernel(
    x_ptr,
    out_ptr,
    scalar,
    alpha,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    tl.store(out_ptr + offs, x + scalar * alpha, mask=mask)


@triton.jit(do_not_specialize=["scalar", "n_elements"])
def _add_tensor_scalar_tiny_alpha1_kernel(
    x_ptr,
    out_ptr,
    scalar,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    tl.store(out_ptr + offs, x + scalar, mask=mask)


@triton.jit(do_not_specialize=["alpha", "last_dim"])
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


@triton.jit(do_not_specialize=["alpha", "total_elements"])
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


@triton.jit(do_not_specialize=["alpha", "total_elements"])
def _add_contiguous_single_program_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    alpha,
    total_elements,
    BLOCK_SIZE: tl.constexpr,
):
    offs = tl.arange(0, BLOCK_SIZE)
    for base in range(0, total_elements, BLOCK_SIZE):
        idx = base + offs
        mask = idx < total_elements
        x = tl.load(x_ptr + idx, mask=mask, other=0.0)
        y = tl.load(y_ptr + idx, mask=mask, other=0.0)
        tl.store(out_ptr + idx, x + y * alpha, mask=mask)


@triton.jit(do_not_specialize=["alpha"])
def _add_contiguous_1024_hot_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    alpha,
):
    offs = tl.arange(0, 256)
    for base in range(0, 1024, 256):
        x = tl.load(x_ptr + base + offs)
        y = tl.load(y_ptr + base + offs)
        tl.store(out_ptr + base + offs, x + y * alpha)


@triton.jit
def _add_contiguous_1024_hot_alpha1_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
):
    offs = tl.arange(0, 256)
    for base in range(0, 1024, 256):
        x = tl.load(x_ptr + base + offs)
        y = tl.load(y_ptr + base + offs)
        tl.store(out_ptr + base + offs, x + y)


@triton.jit(do_not_specialize=["alpha"])
def _add_contiguous_2048_hot_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    alpha,
):
    offs = tl.arange(0, 256)
    for base in range(0, 2048, 256):
        x = tl.load(x_ptr + base + offs)
        y = tl.load(y_ptr + base + offs)
        tl.store(out_ptr + base + offs, x + y * alpha)


@triton.jit
def _add_contiguous_2048_hot_alpha1_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
):
    offs = tl.arange(0, 256)
    for base in range(0, 2048, 256):
        x = tl.load(x_ptr + base + offs)
        y = tl.load(y_ptr + base + offs)
        tl.store(out_ptr + base + offs, x + y)


@triton.jit(do_not_specialize=["alpha"])
def _add_contiguous_3584_hot_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    alpha,
):
    offs = tl.arange(0, 256)
    for base in range(0, 3584, 256):
        x = tl.load(x_ptr + base + offs)
        y = tl.load(y_ptr + base + offs)
        tl.store(out_ptr + base + offs, x + y * alpha)


@triton.jit
def _add_contiguous_3584_hot_alpha1_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
):
    offs = tl.arange(0, 256)
    for base in range(0, 3584, 256):
        x = tl.load(x_ptr + base + offs)
        y = tl.load(y_ptr + base + offs)
        tl.store(out_ptr + base + offs, x + y)


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
    ):
        total = A.numel()
        out = torch.empty_like(A)
        alpha_cast = float(alpha)
        block_size = _select_contiguous_block(total, A.dtype, A)
        _launch_contiguous_add_kernel(A, B, out, alpha_cast, total, block_size)
        return out

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
    ):
        total = A.numel()
        alpha_cast = float(alpha)
        block_size = _select_contiguous_block(total, A.dtype, A)
        _launch_contiguous_add_kernel(A, B, A, alpha_cast, total, block_size)
        return A

    if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        return _arm_add_func(A, B, alpha, out0=A)
    if isinstance(A, torch.Tensor):
        return _arm_add_func_tensor_scalar(A, B, alpha, out0=A)
    raise ValueError("Unreachable.")


def _is_contiguous_add_3584_hotshape(A, B, alpha):
    return (
        alpha == 1
        and isinstance(A, torch.Tensor)
        and isinstance(B, torch.Tensor)
        and A.device.type == "cpu"
        and B.device == A.device
        and A.dtype == B.dtype
        and A.dtype in (torch.bfloat16, torch.float32, torch.float64)
        and A.is_contiguous()
        and B.is_contiguous()
        and A.ndim == 3
        and A.shape == (1, 1, 3584)
        and B.shape == A.shape
    )


def _use_triton(n_elements):
    return n_elements >= 4096


def _use_triton_scalar(n_elements):
    return n_elements >= 1


def _select_contiguous_block(n_elements, dtype, tensor=None):
    if n_elements <= 32:
        return 32
    if tensor is not None:
        # Decode hotspot: [1, 8/16, 1, 128]
        if tensor.ndim == 4 and tensor.shape[2] == 1 and tensor.shape[3] == 128:
            if tensor.shape[1] == 8:
                return 64
            if tensor.shape[1] == 16:
                return 32
        # Decode hotspot: [1, 1, 1024]
        if tensor.ndim == 3 and tensor.shape[1] == 1 and tensor.shape[2] == 1024:
            return 128
    if n_elements <= 2048:
        return 128
    return 256 if dtype in (torch.float16, torch.bfloat16) else 128


def _single_program_block(total_elements):
    if total_elements <= 256:
        return 32
    if total_elements <= 2048:
        return 128
    return 256


def _launch_contiguous_add_kernel(x, y, out, alpha, total_elements, block_size):
    if total_elements == 1024:
        if alpha == 1.0:
            _add_contiguous_1024_hot_alpha1_kernel[(1,)](
                x,
                y,
                out,
                num_warps=1,
                num_stages=1,
            )
            return
        _add_contiguous_1024_hot_kernel[(1,)](
            x,
            y,
            out,
            alpha,
            num_warps=1,
            num_stages=1,
        )
        return
    if total_elements == 2048:
        if alpha == 1.0:
            _add_contiguous_2048_hot_alpha1_kernel[(1,)](
                x,
                y,
                out,
                num_warps=1,
                num_stages=1,
            )
            return
        _add_contiguous_2048_hot_kernel[(1,)](
            x,
            y,
            out,
            alpha,
            num_warps=1,
            num_stages=1,
        )
        return
    if total_elements == 3584:
        if alpha == 1.0:
            _add_contiguous_3584_hot_alpha1_kernel[(1,)](
                x,
                y,
                out,
                num_warps=1,
                num_stages=1,
            )
            return
        _add_contiguous_3584_hot_kernel[(1,)](
            x,
            y,
            out,
            alpha,
            num_warps=1,
            num_stages=1,
        )
        return
    # Small decode tensors benefit from a single-program kernel on triton-cpu.
    if 1 < total_elements <= 16384:
        single_block = _single_program_block(total_elements)
        _add_contiguous_single_program_kernel[(1,)](
            x,
            y,
            out,
            alpha,
            total_elements,
            BLOCK_SIZE=single_block,
            num_warps=1,
            num_stages=1,
        )
        return

    grid = lambda META: (triton.cdiv(total_elements, META["BLOCK_SIZE"]),)
    _add_contiguous_kernel[grid](
        x,
        y,
        out,
        alpha,
        total_elements,
        BLOCK_SIZE=block_size,
        num_warps=1,
        num_stages=1,
    )


def _maybe_contiguous(x, y, out):
    if x.is_contiguous() and (y is None or y.is_contiguous()):
        return x, y, out, False
    x = x if x.is_contiguous() else x.contiguous()
    y = None if y is None else (y if y.is_contiguous() else y.contiguous())
    if out is not None and not out.is_contiguous():
        out = None
    return x, y, out, True


def _launch_scalar_add_kernel(x_contig, out_contig, scalar, alpha, n_elements, block_size):
    if n_elements <= 32:
        if alpha == 1.0:
            _add_tensor_scalar_tiny_alpha1_kernel[(1,)](
                x_contig,
                out_contig,
                scalar,
                n_elements,
                BLOCK_SIZE=32,
                num_warps=1,
                num_stages=1,
            )
            return
        _add_tensor_scalar_tiny_kernel[(1,)](
            x_contig,
            out_contig,
            scalar,
            alpha,
            n_elements,
            BLOCK_SIZE=32,
            num_warps=1,
            num_stages=1,
        )
        return

    if 1 < n_elements <= 16384:
        single_block = _single_program_block(n_elements)
        _add_tensor_scalar_single_program_kernel[(1,)](
            x_contig,
            out_contig,
            scalar,
            alpha,
            n_elements,
            BLOCK_SIZE=single_block,
            num_warps=1,
            num_stages=1,
        )
        return

    grid = (triton.cdiv(n_elements, block_size),)
    _add_tensor_scalar_kernel[grid](
        x_contig,
        out_contig,
        scalar,
        alpha,
        n_elements,
        BLOCK_SIZE=block_size,
        num_warps=1,
        num_stages=1,
    )


def _add_tensor_scalar_triton(x, scalar, alpha, out=None):
    n_elements = x.numel()
    if n_elements == 0:
        return x if out is None else out
    if n_elements == 1:
        val = float(x.item()) + float(scalar) * float(alpha)
        if out is None:
            out = torch.empty_like(x)
        out.fill_(val)
        return out
    block_size = _select_contiguous_block(n_elements, x.dtype, x)
    x_contig = x if x.is_contiguous() else x.contiguous()
    out_contig = out
    if out_contig is None:
        out_contig = torch.empty_like(x_contig)
    _launch_scalar_add_kernel(x_contig, out_contig, scalar, alpha, n_elements, block_size)
    return out_contig


def _add_tensor_tensor_triton(x, y, alpha, out=None):
    n_elements = x.numel()
    if n_elements == 0:
        return x if out is None else out
    x_contig, y_contig, out_contig, _ = _maybe_contiguous(x, y, out)
    if out_contig is None:
        out_contig = torch.empty_like(x_contig)
    if (
        x_contig.shape == y_contig.shape
        and x_contig.is_contiguous()
        and y_contig.is_contiguous()
    ):
        block_size = _select_contiguous_block(n_elements, x_contig.dtype, x_contig)
        _launch_contiguous_add_kernel(
            x_contig, y_contig, out_contig, float(alpha), n_elements, block_size
        )
        return out_contig
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
        block_size = _select_contiguous_block(last_dim, x_contig.dtype, x_contig)
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


def _is_hotshape_tensor_tensor_add(x, y):
    if not (
        isinstance(x, torch.Tensor)
        and isinstance(y, torch.Tensor)
        and x.device.type == "cpu"
        and y.device == x.device
        and x.is_contiguous()
        and y.is_contiguous()
        and x.shape == y.shape
        and x.dtype == y.dtype
        and x.dtype in (torch.bfloat16, torch.float32, torch.float64)
    ):
        return False
    return x.numel() in (1024, 2048, 2560, 5120)


def _add_tensor_tensor_hotshape_triton(x, y, alpha, out=None):
    if not _is_hotshape_tensor_tensor_add(x, y):
        return None
    if out is not None and not out.is_contiguous():
        return None
    n_elements = x.numel()
    out_tensor = torch.empty_like(x) if out is None else out
    block_size = _select_contiguous_block(n_elements, x.dtype, x)
    _launch_contiguous_add_kernel(
        x, y, out_tensor, float(alpha), n_elements, block_size
    )
    return out_tensor


def _add_tensor_tensor_3d_lastdim1024(x, y, alpha, out=None):
    n_elements = x.numel()
    if n_elements == 0:
        return x if out is None else out
    x_contig, y_contig, out_contig, _ = _maybe_contiguous(x, y, out)
    if out_contig is None:
        out_contig = torch.empty_like(x_contig)
    block_size = _select_contiguous_block(n_elements, x_contig.dtype, x_contig)
    _launch_contiguous_add_kernel(
        x_contig, y_contig, out_contig, alpha, n_elements, block_size
    )
    return out_contig


def _maybe_prewarm_add_kernels():
    global _PREWARM_ADD_DONE
    if _PREWARM_ADD_DONE:
        return
    if not _ADD_PREWARM_ENABLED:
        _PREWARM_ADD_DONE = True
        return
    try:
        for dt in (torch.float32, torch.bfloat16):
            x = torch.zeros((1, 1, 1024), dtype=dt, device="cpu")
            y = torch.zeros((1, 1, 1024), dtype=dt, device="cpu")
            out = torch.empty_like(x)
            block = _select_contiguous_block(x.numel(), x.dtype, x)
            _launch_contiguous_add_kernel(x, y, out, 1.0, x.numel(), block)

            ys = torch.zeros((1, 1, 1), dtype=dt, device="cpu")
            _add_broadcast_lastdim_kernel[(triton.cdiv(x.shape[-1], block),)](
                x,
                ys,
                out,
                1.0,
                x.shape[-1],
                triton.cdiv(x.shape[-1], block),
                BLOCK_SIZE=block,
                num_warps=1,
                num_stages=1,
            )

            _launch_scalar_add_kernel(x, out, 1.0, 1.0, x.numel(), block)

            # Decode hotspot: [1, 16, 1, 128] contiguous add
            x2048 = torch.zeros((1, 16, 1, 128), dtype=dt, device="cpu")
            y2048 = torch.zeros((1, 16, 1, 128), dtype=dt, device="cpu")
            out2048 = torch.empty_like(x2048)
            block2048 = _select_contiguous_block(x2048.numel(), x2048.dtype, x2048)
            _launch_contiguous_add_kernel(
                x2048,
                y2048,
                out2048,
                1.0,
                x2048.numel(),
                block2048,
            )

            # Decode hotspot: [1, 1, 3584] contiguous add
            x3584 = torch.zeros((1, 1, 3584), dtype=dt, device="cpu")
            y3584 = torch.zeros((1, 1, 3584), dtype=dt, device="cpu")
            out3584 = torch.empty_like(x3584)
            block3584 = _select_contiguous_block(x3584.numel(), x3584.dtype, x3584)
            _launch_contiguous_add_kernel(
                x3584,
                y3584,
                out3584,
                1.0,
                x3584.numel(),
                block3584,
            )

            # Decode hotspot scalar-add shapes.
            x1 = torch.zeros((1, 1, 1), dtype=dt, device="cpu")
            out1 = torch.empty_like(x1)
            _launch_scalar_add_kernel(x1, out1, 1.0, 1.0, x1.numel(), 32)

            x8 = torch.zeros((1, 1, 8, 1), dtype=dt, device="cpu")
            out8 = torch.empty_like(x8)
            _launch_scalar_add_kernel(x8, out8, 1.0, 1.0, x8.numel(), 32)
    except Exception:
        logging.debug("GEMS ARM add prewarm failed", exc_info=True)
    _PREWARM_ADD_DONE = True


def add(A, B, *, alpha=1):
    logging.debug("GEMS_ARM ADD")
    if _ADD_TRITON_ENABLED and _is_contiguous_add_3584_hotshape(A, B, alpha):
        out = torch.empty_like(A)
        _add_contiguous_3584_hot_alpha1_kernel[(1,)](
            A,
            B,
            out,
            num_warps=1,
            num_stages=1,
        )
        return out
    _maybe_prewarm_add_kernels()
    if not _ADD_TRITON_ENABLED:
        return _base_add(A, B, alpha=alpha)
    if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        hot = _add_tensor_tensor_hotshape_triton(A, B, alpha)
        if hot is not None:
            return hot
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
        if A.shape == B.shape and A.is_contiguous() and B.is_contiguous():
            return _add_tensor_tensor_triton(A, B, alpha)
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
    if _ADD_TRITON_ENABLED and _is_contiguous_add_3584_hotshape(A, B, alpha):
        _add_contiguous_3584_hot_alpha1_kernel[(1,)](
            A,
            B,
            A,
            num_warps=1,
            num_stages=1,
        )
        return A
    _maybe_prewarm_add_kernels()
    if not _ADD_TRITON_ENABLED:
        return _base_add_(A, B, alpha=alpha)
    if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        hot = _add_tensor_tensor_hotshape_triton(A, B, alpha, out=A)
        if hot is not None:
            return hot
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
        if A.shape == B.shape and A.is_contiguous() and B.is_contiguous():
            return _add_tensor_tensor_triton(A, B, alpha, out=A)
        if _use_triton(A.numel()) and _is_broadcast_lastdim(A, B):
            return _add_tensor_tensor_triton(A, B, alpha, out=A)
        if _use_triton(A.numel()) and _is_lastdim1024_3d(A, B):
            return _add_tensor_tensor_3d_lastdim1024(A, B, alpha, out=A)
        if alpha == 1 and _use_triton(B.numel()) and _is_broadcast_lastdim(B, A):
            return _add_tensor_tensor_triton(B, A, alpha, out=A)
        return _base_add_(A, B, alpha=alpha)
    return _base_add_(A, B, alpha=alpha)


_maybe_prewarm_add_kernels()
