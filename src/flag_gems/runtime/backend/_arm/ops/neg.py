import logging
import os

import torch
import triton
import triton.language as tl

from flag_gems.ops import neg as base_neg
from flag_gems.utils import pointwise_dynamic
from flag_gems.utils.codegen_config_utils import CodeGenConfig

_ARM_NEG_CONFIG = CodeGenConfig(
    max_tile_size=256,
    max_grid_size=(2147483647, 1, 1),
    max_num_warps_per_cta=1,
    prefer_block_pointer=False,
    prefer_1d_tile=True,
)


@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")], config=_ARM_NEG_CONFIG)
@triton.jit
def _neg_pointwise(x):
    return -x


@triton.jit
def _neg_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    num_prog = tl.num_programs(0)
    start = pid * BLOCK_SIZE
    step = num_prog * BLOCK_SIZE
    for off in range(start, n_elements, step):
        offsets = off + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        y = -x
        tl.store(out_ptr + offsets, y, mask=mask)


def _select_block_size(n_elements, dtype):
    if n_elements >= (1 << 20):
        return 256 if dtype in (torch.float16, torch.bfloat16) else 128
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


def _neg_triton_custom(x, out=None):
    n_elements = x.numel()
    if n_elements == 0:
        return x if out is None else out
    if os.environ.get("GEMS_DEBUG_NEG") == "1":
        print(f"[GEMS_DEBUG_NEG] _neg_triton_custom: shape={tuple(x.shape)} dtype={x.dtype}")
    block_size = _select_block_size(n_elements, x.dtype)
    block_size = min(block_size, triton.next_power_of_2(max(n_elements, 1)))
    num_blocks = triton.cdiv(n_elements, block_size)
    grid = (num_blocks,)
    x_contig, out_contig, _ = _maybe_contiguous(x, out)
    if out_contig is None:
        out_contig = torch.empty_like(x_contig)
    _neg_kernel[grid](
        x_contig,
        out_contig,
        n_elements,
        BLOCK_SIZE=block_size,
        num_warps=1,
    )
    return out_contig


def _neg_dispatch_tensor(A, out=None):
    # bfloat16 scalar hits a Triton-CPU LLVM issue; do a tiny Python fallback.
    if A.dtype is torch.bfloat16 and A.numel() == 1:
        val = -A.item()
        if out is None:
            out = torch.empty_like(A)
        out.fill_(val)
        return out
    # bfloat16 hits a Triton-CPU LLVM issue with the custom kernel; use pointwise_dynamic for it.
    if A.dtype is torch.bfloat16:
        return _neg_pointwise(A) if out is None else _neg_pointwise(A, out0=out)
    return _neg_triton_custom(A, out=out)


def neg(A):
    logging.debug("GEMS_ARM NEG")
    if isinstance(A, torch.Tensor):
        return _neg_dispatch_tensor(A)
    return base_neg.neg(A)


def neg_(A):
    logging.debug("GEMS_ARM NEG_")
    if isinstance(A, torch.Tensor):
        if A.is_contiguous():
            return _neg_dispatch_tensor(A, out=A)
        result = _neg_dispatch_tensor(A, out=None)
        A.copy_(result)
        return A
    return base_neg.neg_(A)
