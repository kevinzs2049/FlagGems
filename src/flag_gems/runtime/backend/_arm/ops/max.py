import logging
import math
from collections import namedtuple

import torch
import triton
import triton.language as tl

from flag_gems import runtime

# from ..runtime import torch_device_fn
# from ..utils import libentry
from flag_gems.utils import triton_lang_extension as tle
from .argmax import argmax_kernel_1


# @libentry()
@triton.jit
def max_kernel_1(
    inp,
    mid,
    M,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    inp_ptrs = inp + offset
    mask = offset < M
    inp_val = tl.load(inp_ptrs, mask=mask, other=-float("inf"))
    max_val = tl.max(inp_val)
    mid_ptr = mid + pid
    tl.store(mid_ptr, max_val)


# @libentry()
@triton.jit
def max_kernel_2(mid, out, mid_size, BLOCK_MID: tl.constexpr):
    offset = tl.arange(0, BLOCK_MID)
    mid_ptrs = mid + offset
    mask = offset < mid_size
    mid_val = tl.load(mid_ptrs, mask=mask, other=-float("inf"))
    max_val = tl.max(mid_val)
    tl.store(out, max_val)


@triton.jit
def max_argmax_kernel_2(
    mid_value,
    mid_index,
    out_value,
    out_index,
    mid_size,
    BLOCK_MID: tl.constexpr,
):
    offset = tl.arange(0, BLOCK_MID)
    mask = offset < mid_size
    mid_val = tl.load(mid_value + offset, mask=mask, other=-float("inf"))
    index_val = tl.argmax(mid_val, axis=0)
    out_val = tl.load(mid_value + index_val)
    out_idx = tl.load(mid_index + index_val)
    tl.store(out_value, out_val)
    tl.store(out_index, out_idx)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 1}, num_warps=1),
        triton.Config({"BLOCK_SIZE": 8}, num_warps=1),
        triton.Config({"BLOCK_SIZE": 2}, num_warps=2),
        triton.Config({"BLOCK_SIZE": 16}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 32}, num_warps=4),
    ],
    key=["M"],  # 当张量大小变化时触发调优
)
# @libentry()
@triton.jit
def max_kernel_3(inp, out, M, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    start = pid * BLOCK_SIZE
    offsets = start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < M
    x = tl.load(inp + offsets, mask=mask)
    min_val = tl.max(x, axis=None)
    tl.atomic_max(out, min_val)


def heur_block_n(args):
    return triton.next_power_of_2(args["N"])


# @libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("max"),
    key=[
        "M",
        "N",
    ],
)
@triton.jit
def max_kernel(
    inp,
    out_value,
    out_index,
    M,
    N,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # set offset
    pid_m = tle.program_id(0)
    pid_k = tle.program_id(1)
    m_offset = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    result_value = tl.full([BLOCK_M], value=-float("inf"), dtype=tl.float32)
    result_index = tl.zeros([BLOCK_M], dtype=tl.int64)
    for i in range(0, N, BLOCK_N):
        n_offset = i + tl.arange(0, BLOCK_N)
        offset = m_offset[:, None] * N * K + n_offset[None, :] * K + pid_k
        # set mask
        mask = m_offset[:, None] < M and n_offset[None, :] < N
        inp_ptrs = inp + offset
        inp_vals = tl.load(inp_ptrs, mask=mask, other=-float("inf"))
        max_value, max_index = tl.max(inp_vals, axis=1, return_indices=True)
        update_mask = max_value > result_value
        result_value = tl.where(update_mask, max_value, result_value)
        result_index = tl.where(update_mask, i + max_index, result_index)
    mask1 = m_offset < M
    offset_index = m_offset * K + pid_k
    out_value_ptrs = out_value + offset_index
    out_index_ptrs = out_index + offset_index

    tl.store(out_value_ptrs, result_value, mask=mask1)
    tl.store(out_index_ptrs, result_index, mask=mask1)


def max(inp):
    inp = inp.contiguous()
    M = inp.numel()
    block_size = triton.next_power_of_2(math.ceil(math.sqrt(M)))
    mid_size = triton.cdiv(M, block_size)
    block_mid = triton.next_power_of_2(mid_size)

    dtype = inp.dtype
    mid = torch.empty((mid_size,), dtype=dtype, device=inp.device)
    out = torch.empty([], dtype=dtype, device=inp.device)

    # Use two-stage reduction for broader dtype support on Triton CPU.
    max_kernel_1[(mid_size, 1, 1)](inp, mid, M, block_size)
    max_kernel_2[(1, 1, 1)](mid, out, mid_size, block_mid)
    return out


def _max_argmax_m1(inp, out_value, out_index, N):
    block_size = triton.next_power_of_2(math.ceil(math.sqrt(N)))
    mid_size = triton.cdiv(N, block_size)
    block_mid = triton.next_power_of_2(mid_size)
    mid_value = torch.empty((mid_size,), dtype=inp.dtype, device=inp.device)
    mid_index = torch.empty((mid_size,), dtype=torch.int64, device=inp.device)
    argmax_kernel_1[(mid_size, 1, 1)](
        inp,
        mid_value,
        mid_index,
        N,
        block_size,
    )
    max_argmax_kernel_2[(1, 1, 1)](
        mid_value,
        mid_index,
        out_value,
        out_index,
        mid_size,
        block_mid,
    )


def max_dim(inp, dim=None, keepdim=False):
    logging.debug("GEMS MAX DIM")
    assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
    dim = dim % inp.ndim
    shape = inp.shape
    N = shape[dim]
    M = math.prod(shape[:dim])
    K = inp.numel() // M // N

    inp = inp.contiguous()
    shape_list = list(shape)
    shape_list[dim] = 1
    out_value = torch.empty(shape_list, dtype=inp.dtype, device=inp.device)
    out_index = torch.empty(shape_list, dtype=torch.int64, device=inp.device)

    # Decode-heavy max over a single row/vocab uses two-stage reduction
    # to parallelize over N and reduce launch overhead.
    if M == 1 and K == 1:
        _max_argmax_m1(
            inp.reshape(-1),
            out_value.reshape(-1),
            out_index.reshape(-1),
            N,
        )
    else:
        grid = lambda meta: (
            triton.cdiv(M, meta["BLOCK_M"]),
            K,
        )
        max_kernel[grid](
            inp,
            out_value,
            out_index,
            M,
            N,
            K,
        )

    if not keepdim:
        out_value = torch.squeeze(out_value, dim)
        out_index = torch.squeeze(out_index, dim)
    Max_out = namedtuple("max", ["values", "indices"])
    out = Max_out(values=out_value, indices=out_index)
    return out
