import logging
import math
import os

import torch
import triton
import triton.language as tl

from flag_gems.ops.rsqrt import rsqrt as base_rsqrt
from flag_gems.ops.rsqrt import rsqrt_ as base_rsqrt_


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


def _select_block_size(n_elements, dtype):
    if n_elements >= (1 << 20):
        return 512 if dtype in (torch.float16, torch.bfloat16) else 256
    if n_elements >= (1 << 18):
        return 256 if dtype in (torch.float16, torch.bfloat16) else 128
    return 256 if dtype in (torch.float16, torch.bfloat16) else 128


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
    if n_elements == 1 and x.dtype is torch.bfloat16:
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
    if os.environ.get("GEMS_DEBUG_RSQRT") == "1":
        print(f"[GEMS_DEBUG_RSQRT] _rsqrt_triton: shape={tuple(x.shape)} dtype={x.dtype}")
    block_size = _select_block_size(n_elements, x.dtype)
    block_size = min(block_size, triton.next_power_of_2(max(n_elements, 1)))
    num_blocks = triton.cdiv(n_elements, block_size)
    grid = (num_blocks,)
    x_contig, out_contig, _ = _maybe_contiguous(x, out)
    if out_contig is None:
        out_contig = torch.empty_like(x_contig)
    _rsqrt_kernel[grid](
        x_contig,
        out_contig,
        n_elements,
        BLOCK_SIZE=block_size,
        num_warps=1,
    )
    return out_contig


def rsqrt(A):
    logging.debug("GEMS_ARM RSQRT")
    if isinstance(A, torch.Tensor):
        return _rsqrt_triton(A)
    return base_rsqrt(A)


def rsqrt_(A):
    logging.debug("GEMS_ARM RSQRT_")
    if isinstance(A, torch.Tensor) and A.is_contiguous():
        return _rsqrt_triton(A, out=A)
    return base_rsqrt_(A)
