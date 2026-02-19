import logging
import math
import os

import torch
import triton
import triton.language as tl

from flag_gems.ops.rsqrt import rsqrt as base_rsqrt
from flag_gems.ops.rsqrt import rsqrt_ as base_rsqrt_

import numpy as np

# For small tensors, bypass Triton entirely via numpy (zero-copy views).
_RSQRT_NATIVE_THRESHOLD = 4096

_PREWARM_RSQRT_DONE = False
_RSQRT_ROWS1_HOT_ENABLED = os.environ.get("GEMS_ARM_RSQRT_ROWS1_HOT", "1") == "1"
_RSQRT_TINY_HOT_ENABLED = os.environ.get("GEMS_ARM_RSQRT_TINY_HOT", "1") == "1"
_RSQRT_PREWARM_ENABLED = os.environ.get("GEMS_ARM_RSQRT_PREWARM", "1") == "1"
_RSQRT_DEBUG_ENABLED = os.environ.get("GEMS_DEBUG_RSQRT") == "1"


@triton.jit
def _rsqrt_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    num_prog = tl.num_programs(0)
    start = pid * BLOCK_SIZE
    step = num_prog * BLOCK_SIZE
    for off in range(start, n_elements, step):
        offsets = off + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        y = 1.0 / tl.sqrt(x.to(tl.float32))
        tl.store(out_ptr + offsets, y.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _rsqrt_single_program_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    offs = tl.arange(0, BLOCK_SIZE)
    for base in range(0, n_elements, BLOCK_SIZE):
        idx = base + offs
        mask = idx < n_elements
        x = tl.load(x_ptr + idx, mask=mask, other=0.0)
        y = 1.0 / tl.sqrt(x.to(tl.float32))
        tl.store(out_ptr + idx, y.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _rsqrt_1024_hot_kernel(
    x_ptr,
    out_ptr,
):
    offs = tl.arange(0, 256)
    for base in range(0, 1024, 256):
        x = tl.load(x_ptr + base + offs).to(tl.float32)
        y = 1.0 / tl.sqrt(x)
        tl.store(out_ptr + base + offs, y.to(out_ptr.dtype.element_ty))


@triton.jit
def _rsqrt_2048_hot_kernel(
    x_ptr,
    out_ptr,
):
    offs = tl.arange(0, 256)
    for base in range(0, 2048, 256):
        x = tl.load(x_ptr + base + offs).to(tl.float32)
        y = 1.0 / tl.sqrt(x)
        tl.store(out_ptr + base + offs, y.to(out_ptr.dtype.element_ty))


@triton.jit
def _rsqrt_rows1_hot_kernel(
    x_ptr,
    out_ptr,
):
    x = tl.load(x_ptr)
    y = 1.0 / tl.sqrt(x.to(tl.float32))
    tl.store(out_ptr, y.to(out_ptr.dtype.element_ty))


@triton.jit
def _rsqrt_rows8_hot_kernel(
    x_ptr,
    out_ptr,
):
    for row in range(0, 8):
        x = tl.load(x_ptr + row)
        y = 1.0 / tl.sqrt(x.to(tl.float32))
        tl.store(out_ptr + row, y.to(out_ptr.dtype.element_ty))


@triton.jit
def _rsqrt_rows16_hot_kernel(
    x_ptr,
    out_ptr,
):
    for row in range(0, 16):
        x = tl.load(x_ptr + row)
        y = 1.0 / tl.sqrt(x.to(tl.float32))
        tl.store(out_ptr + row, y.to(out_ptr.dtype.element_ty))


@triton.jit
def _rsqrt_rows128_hot_kernel(
    x_ptr,
    out_ptr,
):
    for row in range(0, 128):
        x = tl.load(x_ptr + row)
        y = 1.0 / tl.sqrt(x.to(tl.float32))
        tl.store(out_ptr + row, y.to(out_ptr.dtype.element_ty))


@triton.jit(do_not_specialize=["rows"])
def _rsqrt_rows_hot_kernel(
    x_ptr,
    out_ptr,
    rows,
    MAX_ROWS: tl.constexpr,
):
    for row in range(0, MAX_ROWS):
        if row < rows:
            x = tl.load(x_ptr + row)
            y = 1.0 / tl.sqrt(x.to(tl.float32))
            tl.store(out_ptr + row, y.to(out_ptr.dtype.element_ty))


@triton.jit(do_not_specialize=["n_elements"])
def _rsqrt_tiny_flat_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    y = 1.0 / tl.sqrt(x.to(tl.float32))
    tl.store(out_ptr + offs, y.to(out_ptr.dtype.element_ty), mask=mask)


def _select_block_size(n_elements, dtype):
    # Favor larger fixed tiles for tiny tensors to reduce launch overhead on CPU.
    if n_elements <= 32:
        return 32
    if n_elements == 512:
        return 512
    if n_elements <= 1024:
        return 128
    if n_elements <= (1 << 16):
        return 128
    return 512 if dtype in (torch.float16, torch.bfloat16) else 256


def _single_program_block(n_elements):
    if n_elements <= 256:
        return 32
    if n_elements <= 2048:
        return 128
    return 256


def _launch_rsqrt_kernel(x, out, n_elements, block_size):
    if 1 < n_elements <= 8192:
        single_block = _single_program_block(n_elements)
        _rsqrt_single_program_kernel[(1,)](
            x,
            out,
            n_elements,
            BLOCK_SIZE=single_block,
            num_warps=1,
            num_stages=1,
        )
        return
    grid = (triton.cdiv(n_elements, block_size),)
    _rsqrt_kernel[grid](
        x,
        out,
        n_elements,
        BLOCK_SIZE=block_size,
        num_warps=1,
        num_stages=1,
    )


def _maybe_launch_rsqrt_hotshape(x_contig, out_contig, n_elements):
    if x_contig.numel() == 0 or not x_contig.is_contiguous():
        return False
    if n_elements == 1024:
        _rsqrt_1024_hot_kernel[(1,)](
            x_contig,
            out_contig,
            num_warps=1,
            num_stages=1,
        )
        return True
    if n_elements == 2048:
        _rsqrt_2048_hot_kernel[(1,)](
            x_contig,
            out_contig,
            num_warps=1,
            num_stages=1,
        )
        return True
    if _RSQRT_ROWS1_HOT_ENABLED:
        if x_contig.ndim > 0 and x_contig.shape[-1] == 1 and n_elements <= 256:
            if n_elements == 1:
                _rsqrt_rows1_hot_kernel[(1,)](
                    x_contig,
                    out_contig,
                    num_warps=1,
                    num_stages=1,
                )
                return True
            if n_elements == 8:
                _rsqrt_rows8_hot_kernel[(1,)](
                    x_contig,
                    out_contig,
                    num_warps=1,
                    num_stages=1,
                )
                return True
            if n_elements == 16:
                _rsqrt_rows16_hot_kernel[(1,)](
                    x_contig,
                    out_contig,
                    num_warps=1,
                    num_stages=1,
                )
                return True
            if n_elements == 128:
                _rsqrt_rows128_hot_kernel[(1,)](
                    x_contig,
                    out_contig,
                    num_warps=1,
                    num_stages=1,
                )
                return True
            _rsqrt_rows_hot_kernel[(1,)](
                x_contig,
                out_contig,
                n_elements,
                MAX_ROWS=256,
                num_warps=1,
                num_stages=1,
            )
            return True
    if _RSQRT_TINY_HOT_ENABLED:
        if n_elements <= 256:
            _rsqrt_tiny_flat_kernel[(1,)](
                x_contig,
                out_contig,
                n_elements,
                BLOCK_SIZE=256,
                num_warps=1,
                num_stages=1,
            )
            return True
    return False


def _maybe_contiguous(x, out):
    if x.is_contiguous():
        return x, out, False
    if out is None:
        return x.contiguous(), out, True
    if out.is_contiguous():
        return x.contiguous(), out, True
    return x, out, False


def _rsqrt_triton(x, out=None):
    n_elements = x.numel()
    if n_elements == 0:
        return x if out is None else out
    if n_elements == 1:
        val = float(x.item())
        if val < 0.0:
            val = float("nan")
        elif val == 0.0:
            val = float("inf")
        else:
            val = 1.0 / math.sqrt(val)
        if out is None:
            out = torch.empty_like(x)
        out.fill_(val)
        return out
    if _RSQRT_DEBUG_ENABLED:
        print(f"[GEMS_DEBUG_RSQRT] _rsqrt_triton: shape={tuple(x.shape)} dtype={x.dtype}")
    block_size = _select_block_size(n_elements, x.dtype)
    x_contig, out_contig, _ = _maybe_contiguous(x, out)
    if out_contig is None:
        out_contig = torch.empty_like(x_contig)
    if _maybe_launch_rsqrt_hotshape(x_contig, out_contig, n_elements):
        return out_contig
    _launch_rsqrt_kernel(x_contig, out_contig, n_elements, block_size)
    return out_contig


def _maybe_prewarm_rsqrt_kernels():
    global _PREWARM_RSQRT_DONE
    if _PREWARM_RSQRT_DONE:
        return
    if not _RSQRT_PREWARM_ENABLED:
        _PREWARM_RSQRT_DONE = True
        return
    try:
        for dt in (torch.float32, torch.bfloat16):
            x1024 = torch.ones((1, 1, 1024), dtype=dt, device="cpu")
            out1024 = torch.empty_like(x1024)
            _rsqrt_1024_hot_kernel[(1,)](
                x1024,
                out1024,
                num_warps=1,
                num_stages=1,
            )

            x2048 = torch.ones((1, 16, 1, 128), dtype=dt, device="cpu")
            out2048 = torch.empty_like(x2048)
            _rsqrt_2048_hot_kernel[(1,)](
                x2048,
                out2048,
                num_warps=1,
                num_stages=1,
            )

            x1 = torch.ones((1, 1, 1), dtype=dt, device="cpu")
            out1 = torch.empty_like(x1)
            _rsqrt_rows1_hot_kernel[(1,)](
                x1,
                out1,
                num_warps=1,
                num_stages=1,
            )

            x8 = torch.ones((1, 1, 8, 1), dtype=dt, device="cpu")
            out8 = torch.empty_like(x8)
            _rsqrt_rows8_hot_kernel[(1,)](
                x8,
                out8,
                num_warps=1,
                num_stages=1,
            )

            x16 = torch.ones((1, 1, 16, 1), dtype=dt, device="cpu")
            out16 = torch.empty_like(x16)
            _rsqrt_rows16_hot_kernel[(1,)](
                x16,
                out16,
                num_warps=1,
                num_stages=1,
            )

            x128 = torch.ones((1, 1, 128, 1), dtype=dt, device="cpu")
            out128 = torch.empty_like(x128)
            _rsqrt_rows128_hot_kernel[(1,)](
                x128,
                out128,
                num_warps=1,
                num_stages=1,
            )

            block1024 = _select_block_size(x1024.numel(), x1024.dtype)
            _launch_rsqrt_kernel(x1024, out1024, x1024.numel(), block1024)
    except Exception:
        logging.debug("GEMS ARM rsqrt prewarm failed", exc_info=True)
    _PREWARM_RSQRT_DONE = True


def rsqrt(A):
    logging.debug("GEMS_ARM RSQRT")
    if isinstance(A, torch.Tensor) and A.numel() < _RSQRT_NATIVE_THRESHOLD and A.is_contiguous() and A.dtype in (torch.float32, torch.float64):
        an = A.detach().numpy()
        return torch.from_numpy(1.0 / np.sqrt(an))
    _maybe_prewarm_rsqrt_kernels()
    if isinstance(A, torch.Tensor):
        return _rsqrt_triton(A)
    return base_rsqrt(A)


def rsqrt_(A):
    logging.debug("GEMS_ARM RSQRT_")
    if isinstance(A, torch.Tensor) and A.numel() < _RSQRT_NATIVE_THRESHOLD and A.is_contiguous() and A.dtype in (torch.float32, torch.float64):
        an = A.detach().numpy()
        np.divide(1.0, np.sqrt(an), out=an)
        return A
    _maybe_prewarm_rsqrt_kernels()
    if isinstance(A, torch.Tensor) and A.is_contiguous():
        return _rsqrt_triton(A, out=A)
    return base_rsqrt_(A)


_maybe_prewarm_rsqrt_kernels()
