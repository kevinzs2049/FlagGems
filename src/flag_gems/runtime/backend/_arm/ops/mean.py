import logging
import math
import os

import torch
import triton
import triton.language as tl

from flag_gems import runtime

# from ..utils import dim_compress, libentry
# from ..runtime import torch_device_fn
from flag_gems.utils import dim_compress
from flag_gems.utils import triton_lang_extension as tle

_PREWARM_MEAN_DONE = False


# @libentry()
@triton.jit
def mean_kernel_1(
    inp,
    mid,
    M,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    inp_ptrs = inp + offset
    mask = offset < M
    inp_val = tl.load(inp_ptrs, mask=mask, other=0.0)
    sum_val = tl.sum(inp_val, axis=0)
    mid_ptr = mid + pid
    tl.store(mid_ptr, sum_val)


# @libentry()
@triton.jit
def mean_kernel_2(mid, out, M, MID_SIZE, BLOCK_MID: tl.constexpr):
    offset = tl.arange(0, BLOCK_MID)
    mid_ptrs = mid + offset
    mask = offset < MID_SIZE
    mid_val = tl.load(mid_ptrs, mask=mask, other=0.0)
    sum_val = tl.sum(mid_val, axis=0) / M
    tl.store(out, sum_val)


@triton.jit(do_not_specialize=["rows", "cols"])
def _mean_lastdim_fast_kernel(inp, out, rows, cols, BLOCK_SIZE: tl.constexpr):
    row = tle.program_id(0)
    if row >= rows:
        return
    offs = tl.arange(0, BLOCK_SIZE)
    row_ptr = inp + row * cols
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for base in range(0, cols, BLOCK_SIZE):
        idx = base + offs
        mask = idx < cols
        x = tl.load(row_ptr + idx, mask=mask, other=0.0).to(tl.float32)
        acc += x
    mean_val = tl.sum(acc, axis=0) / cols
    tl.store(out + row, mean_val.to(out.dtype.element_ty))


@triton.jit
def _mean_lastdim_1x1024_hot_kernel(inp, out):
    offs = tl.arange(0, 256)
    acc = 0.0
    for base in range(0, 1024, 256):
        x = tl.load(inp + base + offs).to(tl.float32)
        acc += tl.sum(x, axis=0)
    tl.store(out, (acc / 1024.0).to(out.dtype.element_ty))


@triton.jit(do_not_specialize=["rows"])
def _mean_lastdim_rows128_hot_kernel(inp, out, rows, MAX_ROWS: tl.constexpr):
    offs = tl.arange(0, 128)
    for row in range(0, MAX_ROWS):
        if row < rows:
            x = tl.load(inp + row * 128 + offs).to(tl.float32)
            mean_val = tl.sum(x, axis=0) / 128.0
            tl.store(out + row, mean_val.to(out.dtype.element_ty))


@triton.jit
def _mean_lastdim_16x128_hot_kernel(inp, out):
    offs = tl.arange(0, 128)
    for row in range(0, 16):
        x = tl.load(inp + row * 128 + offs).to(tl.float32)
        mean_val = tl.sum(x, axis=0) / 128.0
        tl.store(out + row, mean_val.to(out.dtype.element_ty))


@triton.jit
def _mean_lastdim_8x128_hot_kernel(inp, out):
    offs = tl.arange(0, 128)
    for row in range(0, 8):
        x = tl.load(inp + row * 128 + offs).to(tl.float32)
        mean_val = tl.sum(x, axis=0) / 128.0
        tl.store(out + row, mean_val.to(out.dtype.element_ty))


def _supported_fast_mean_dtype(inp_dtype, out_dtype):
    return inp_dtype in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    ) and out_dtype in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    )


def _launch_mean_lastdim_fast(inp_2d, out_1d, rows, cols):
    if cols <= 1024:
        block_size = 128
    else:
        block_size = 256
    _mean_lastdim_fast_kernel[(rows,)](
        inp_2d,
        out_1d,
        rows,
        cols,
        BLOCK_SIZE=block_size,
        num_warps=1,
        num_stages=1,
    )


def _launch_mean_hot_1x1024(inp_2d, out_1d):
    _mean_lastdim_1x1024_hot_kernel[(1,)](
        inp_2d,
        out_1d,
        num_warps=1,
        num_stages=1,
    )


def _launch_mean_hot_rows128(inp_2d, out_1d, rows):
    if rows == 16:
        _mean_lastdim_16x128_hot_kernel[(1,)](
            inp_2d,
            out_1d,
            num_warps=1,
            num_stages=1,
        )
        return
    if rows == 8:
        _mean_lastdim_8x128_hot_kernel[(1,)](
            inp_2d,
            out_1d,
            num_warps=1,
            num_stages=1,
        )
        return
    _mean_lastdim_rows128_hot_kernel[(1,)](
        inp_2d,
        out_1d,
        rows,
        MAX_ROWS=16,
        num_warps=1,
        num_stages=1,
    )


def _maybe_prewarm_mean_kernels():
    global _PREWARM_MEAN_DONE
    if _PREWARM_MEAN_DONE:
        return
    if os.environ.get("GEMS_ARM_MEAN_PREWARM", "1") != "1":
        _PREWARM_MEAN_DONE = True
        return
    try:
        x1024 = torch.zeros((1, 1, 1024), dtype=torch.float32, device="cpu")
        out1024 = torch.empty((1,), dtype=torch.float32, device="cpu")
        _launch_mean_lastdim_fast(x1024.view(1, 1024), out1024, 1, 1024)
        _launch_mean_hot_1x1024(x1024.view(1, 1024), out1024)

        x128 = torch.zeros((1, 1, 16, 128), dtype=torch.float32, device="cpu")
        out128 = torch.empty((16,), dtype=torch.float32, device="cpu")
        _launch_mean_lastdim_fast(x128.view(16, 128), out128, 16, 128)
        _launch_mean_hot_rows128(x128.view(16, 128), out128, 16)
    except Exception:
        logging.debug("GEMS ARM mean prewarm failed", exc_info=True)
    _PREWARM_MEAN_DONE = True


def mean(inp, *, dtype=None):
    logging.debug("GEMS MEAN")
    _maybe_prewarm_mean_kernels()
    M = inp.numel()
    if dtype is None:
        dtype = inp.dtype
    block_size = triton.next_power_of_2(math.ceil(math.sqrt(M)))
    mid_size = triton.cdiv(M, block_size)
    block_mid = triton.next_power_of_2(mid_size)

    mid = torch.empty((mid_size,), dtype=dtype, device=inp.device)
    out = torch.empty([], dtype=dtype, device=inp.device)

    # with torch_device_fn.device(inp.device):
    mean_kernel_1[(mid_size, 1, 1)](inp, mid, M, block_size)
    mean_kernel_2[(1, 1, 1)](mid, out, M, mid_size, block_mid)
    return out


# @libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("mean"),
    key=["M", "N"],
)
@triton.jit
def mean_dim_kernel(X, Mean, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    # Map the program id to the row of X it should compute.
    pid = tle.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    X = X + pid * N
    Mean = Mean + pid
    row_mask = pid < M

    # Compute mean
    _mean = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        col_mask = cols < N
        mask = row_mask and col_mask

        a = tl.load(X + cols, mask, other=0.0).to(tl.float32)
        _mean += a
    mean = tl.sum(_mean, axis=1) / N
    mean = mean[:, None]
    tl.store(Mean, mean, row_mask)


def mean_dim(x, dim=None, keepdim=False, *, dtype=None):
    logging.debug("GEMS MEAN DIM")
    _maybe_prewarm_mean_kernels()

    if dtype is None:
        dtype = x.dtype
    if dim is None:
        out = mean(x, dtype=dtype)
        if keepdim:
            out = out.reshape([1] * x.ndim)
        return out
    elif isinstance(dim, int):
        dim = (dim,)
    elif dim == []:
        out = mean(x, dtype=dtype)
        if keepdim:
            out = out.reshape([1] * x.ndim)
        return out

    shape = list(x.shape)
    dim = [d % x.ndim for d in dim]
    if (
        len(dim) == 1
        and dim[0] == x.ndim - 1
        and x.device.type == "cpu"
        and x.is_contiguous()
        and _supported_fast_mean_dtype(x.dtype, dtype)
    ):
        cols = x.shape[-1]
        rows = x.numel() // cols
        if cols == 1:
            out = x.to(dtype=dtype).clone()
            if not keepdim:
                out = out.squeeze(-1)
            return out
        if rows == 1 and cols == 1024:
            out_flat = torch.empty((1,), dtype=dtype, device=x.device)
            _launch_mean_hot_1x1024(x.view(1, 1024), out_flat)
            out = out_flat.view(*x.shape[:-1], 1)
            if not keepdim:
                out = out.squeeze(-1)
            return out
        if cols == 128 and rows <= 16:
            out_flat = torch.empty((rows,), dtype=dtype, device=x.device)
            _launch_mean_hot_rows128(x.view(rows, 128), out_flat, rows)
            out = out_flat.view(*x.shape[:-1], 1)
            if not keepdim:
                out = out.squeeze(-1)
            return out
        if cols <= 2048:
            out_flat = torch.empty((rows,), dtype=dtype, device=x.device)
            _launch_mean_lastdim_fast(x.view(rows, cols), out_flat, rows, cols)
            out = out_flat.view(*x.shape[:-1], 1)
            if not keepdim:
                out = out.squeeze(-1)
            return out

    x = dim_compress(x, dim)
    N = 1
    for i in dim:
        N *= shape[i]
        shape[i] = 1
    M = x.numel() // N
    out = torch.empty(shape, dtype=dtype, device=x.device)
    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]),)

    # with torch_device_fn.device(x.device):
    mean_dim_kernel[grid](x, out, M, N)
    if not keepdim:
        out = out.squeeze(dim)
    return out


_maybe_prewarm_mean_kernels()
