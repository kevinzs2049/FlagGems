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
_PREWARM_NEG_DONE = False
_NEG_ROWS64_HOT_ENABLED = os.environ.get("GEMS_ARM_NEG_ROWS64_HOT", "1") == "1"
_NEG_PREWARM_ENABLED = os.environ.get("GEMS_ARM_NEG_PREWARM", "0") == "1"
_NEG_DEBUG_ENABLED = os.environ.get("GEMS_DEBUG_NEG") == "1"


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


@triton.jit
def _neg_single_program_kernel(
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
        y = -x
        tl.store(out_ptr + idx, y, mask=mask)


@triton.jit
def _neg_512_hot_kernel(
    x_ptr,
    out_ptr,
):
    offs = tl.arange(0, 256)
    for base in range(0, 512, 256):
        x = tl.load(x_ptr + base + offs)
        tl.store(out_ptr + base + offs, -x)


@triton.jit
def _neg_1024_hot_kernel(
    x_ptr,
    out_ptr,
):
    offs = tl.arange(0, 256)
    for base in range(0, 1024, 256):
        x = tl.load(x_ptr + base + offs)
        tl.store(out_ptr + base + offs, -x)


@triton.jit(do_not_specialize=["rows"])
def _neg_rows64_hot_kernel(
    x_ptr,
    out_ptr,
    rows,
    MAX_ROWS: tl.constexpr,
):
    offs = tl.arange(0, 64)
    for row in range(0, MAX_ROWS):
        if row < rows:
            base = row * 64
            x = tl.load(x_ptr + base + offs)
            tl.store(out_ptr + base + offs, -x)


@triton.jit
def _neg_rows64_8_hot_kernel(
    x_ptr,
    out_ptr,
):
    offs = tl.arange(0, 64)
    for row in range(0, 8):
        base = row * 64
        x = tl.load(x_ptr + base + offs)
        tl.store(out_ptr + base + offs, -x)


@triton.jit
def _neg_rows64_16_hot_kernel(
    x_ptr,
    out_ptr,
):
    offs = tl.arange(0, 64)
    for row in range(0, 16):
        base = row * 64
        x = tl.load(x_ptr + base + offs)
        tl.store(out_ptr + base + offs, -x)


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
    return 256 if dtype in (torch.float16, torch.bfloat16) else 128


def _single_program_block(n_elements):
    if n_elements <= 256:
        return 32
    if n_elements <= 2048:
        return 128
    return 256


def _launch_neg_kernel(x, out, n_elements, block_size):
    if 1 < n_elements <= 8192:
        single_block = _single_program_block(n_elements)
        _neg_single_program_kernel[(1,)](
            x,
            out,
            n_elements,
            BLOCK_SIZE=single_block,
            num_warps=1,
            num_stages=1,
        )
        return
    grid = (triton.cdiv(n_elements, block_size),)
    _neg_kernel[grid](
        x,
        out,
        n_elements,
        BLOCK_SIZE=block_size,
        num_warps=1,
        num_stages=1,
    )


def _maybe_launch_neg_hotshape(x_contig, out_contig):
    if x_contig.numel() == 512:
        _neg_512_hot_kernel[(1,)](
            x_contig,
            out_contig,
            num_warps=1,
            num_stages=1,
        )
        return True
    if x_contig.numel() == 1024:
        _neg_1024_hot_kernel[(1,)](
            x_contig,
            out_contig,
            num_warps=1,
            num_stages=1,
        )
        return True
    if not _NEG_ROWS64_HOT_ENABLED:
        return False
    if x_contig.numel() == 0 or not x_contig.is_contiguous():
        return False
    if x_contig.ndim == 0:
        return False
    if x_contig.shape[-1] != 64:
        return False
    rows = x_contig.numel() // 64
    if rows == 0 or rows > 128 or rows * 64 != x_contig.numel():
        return False
    if rows == 8:
        _neg_rows64_8_hot_kernel[(1,)](
            x_contig,
            out_contig,
            num_warps=1,
            num_stages=1,
        )
        return True
    if rows == 16:
        _neg_rows64_16_hot_kernel[(1,)](
            x_contig,
            out_contig,
            num_warps=1,
            num_stages=1,
        )
        return True
    _neg_rows64_hot_kernel[(1,)](
        x_contig,
        out_contig,
        rows,
        MAX_ROWS=128,
        num_warps=1,
        num_stages=1,
    )
    return True


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
    if _NEG_DEBUG_ENABLED:
        print(f"[GEMS_DEBUG_NEG] _neg_triton_custom: shape={tuple(x.shape)} dtype={x.dtype}")
    block_size = _select_block_size(n_elements, x.dtype)
    x_contig, out_contig, _ = _maybe_contiguous(x, out)
    if out_contig is None:
        out_contig = torch.empty_like(x_contig)
    if _maybe_launch_neg_hotshape(x_contig, out_contig):
        return out_contig
    _launch_neg_kernel(x_contig, out_contig, n_elements, block_size)
    return out_contig


def _maybe_prewarm_neg_kernels():
    global _PREWARM_NEG_DONE
    if _PREWARM_NEG_DONE:
        return
    if not _NEG_PREWARM_ENABLED:
        _PREWARM_NEG_DONE = True
        return
    try:
        for dt in (torch.float32, torch.bfloat16):
            x1024 = torch.zeros((1, 1, 1024), dtype=dt, device="cpu")
            out1024 = torch.empty_like(x1024)
            block1024 = _select_block_size(x1024.numel(), x1024.dtype)
            _launch_neg_kernel(x1024, out1024, x1024.numel(), block1024)

            x128 = torch.zeros((1, 16, 1, 128), dtype=dt, device="cpu")
            out128 = torch.empty_like(x128)
            block128 = _select_block_size(x128.numel(), x128.dtype)
            _launch_neg_kernel(x128, out128, x128.numel(), block128)
    except Exception:
        logging.debug("GEMS ARM neg prewarm failed", exc_info=True)
    _PREWARM_NEG_DONE = True


def _neg_dispatch_tensor(A, out=None):
    # bfloat16 scalar hits a Triton-CPU LLVM issue; do a tiny Python fallback.
    if A.dtype is torch.bfloat16 and A.numel() == 1:
        val = -A.item()
        if out is None:
            out = torch.empty_like(A)
        out.fill_(val)
        return out
    return _neg_triton_custom(A, out=out)


def neg(A):
    logging.debug("GEMS_ARM NEG")
    _maybe_prewarm_neg_kernels()
    if isinstance(A, torch.Tensor):
        return _neg_dispatch_tensor(A)
    return base_neg.neg(A)


def neg_(A):
    logging.debug("GEMS_ARM NEG_")
    _maybe_prewarm_neg_kernels()
    if isinstance(A, torch.Tensor):
        if A.is_contiguous():
            return _neg_dispatch_tensor(A, out=A)
        result = _neg_dispatch_tensor(A, out=None)
        A.copy_(result)
        return A
    return base_neg.neg_(A)


_maybe_prewarm_neg_kernels()
