import logging
import os

import torch
import triton
import triton.language as tl

from flag_gems.utils import triton_lang_extension as tle


MM_GENERIC_CONFIG_TABLE = (
    # Decode-like long vocab projection prefers narrower N tiles.
    {"m_max": 1, "n_min": 65536, "k_min": 0, "config": (4, 16, 8)},
    # Batched decode/prefill small-M cases.
    {"m_max": 8, "n_min": 2048, "k_min": 0, "config": (8, 16, 8)},
    {"m_max": 8, "n_min": 0, "k_min": 2048, "config": (8, 32, 8)},
    {"m_max": 8, "n_min": 0, "k_min": 0, "config": (8, 8, 8)},
)

MM_M1_CONFIG_TABLE = (
    # Keep very large vocab projection on the generic kernel.
    {"n_min": 65536, "k_min": 0, "config": None},
    {"n_min": 2048, "k_min": 0, "config": (32, 8)},
    {"n_min": 0, "k_min": 3072, "config": (128, 8)},
    {"n_min": 0, "k_min": 2048, "config": (32, 16)},
    {"n_min": 0, "k_min": 0, "config": (64, 8)},
)


@triton.jit
def mm_kernel(
    A,
    B,
    C,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    dot_out_dtype: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    SPLIT_K: tl.constexpr,
    EVEN_K: tl.constexpr,
):
    # matrix multiplication
    pid = tle.program_id(0)
    pid_z = tle.program_id(1)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)
    # do matrix multiplication
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    rk = pid_z * BLOCK_K + tl.arange(0, BLOCK_K)
    # pointers
    A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak)
    B = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=dot_out_dtype)
    for k in range(0, tl.cdiv(K, BLOCK_K * SPLIT_K)):
        if EVEN_K:
            a = tl.load(A)
            b = tl.load(B)
        else:
            k_remaining = K - k * (BLOCK_K * SPLIT_K)
            _0 = tl.zeros((1, 1), dtype=C.dtype.element_ty)
            a = tl.load(A, mask=rk[None, :] < k_remaining, other=_0)
            b = tl.load(B, mask=rk[:, None] < k_remaining, other=_0)
        if a.dtype != b.dtype:
            a = a.to(C.dtype.element_ty)
            b = b.to(C.dtype.element_ty)
        acc += tl.dot(a, b, out_dtype=dot_out_dtype, allow_tf32=False)
        A += BLOCK_K * SPLIT_K * stride_ak
        B += BLOCK_K * SPLIT_K * stride_bk
    acc = acc.to(C.dtype.element_ty)
    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    C = C + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    mask = (rm < M)[:, None] & (rn < N)[None, :]
    # handles write-back with reduction-splitting
    if SPLIT_K == 1:
        tl.store(C, acc, mask=mask)
    else:
        tl.atomic_add(C, acc, mask=mask)


@triton.jit
def mm_m1_kernel(
    A,
    B,
    C,
    N,
    K,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cn,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    EVEN_K: tl.constexpr,
):
    pid_n = tle.program_id(0)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)

    a_ptr = A + rk * stride_ak
    b_ptr = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        if EVEN_K:
            a = tl.load(a_ptr)
            b = tl.load(b_ptr)
        else:
            k_remaining = K - k * BLOCK_K
            a = tl.load(a_ptr, mask=rk < k_remaining, other=0.0)
            b = tl.load(
                b_ptr,
                mask=(rk[:, None] < k_remaining) & (rn[None, :] < N),
                other=0.0,
            )

        if a.dtype != b.dtype:
            a = a.to(C.dtype.element_ty)
            b = b.to(C.dtype.element_ty)

        acc += tl.sum(b * a[:, None], axis=0)
        a_ptr += BLOCK_K * stride_ak
        b_ptr += BLOCK_K * stride_bk

    c_ptr = C + rn * stride_cn
    tl.store(c_ptr, acc.to(C.dtype.element_ty), mask=rn < N)


_ordered_datatypes = [torch.float16, torch.bfloat16, torch.float32]


def get_higher_dtype(a, b):
    if a is b:
        return a

    assert a in _ordered_datatypes
    assert b in _ordered_datatypes

    for d in _ordered_datatypes:
        if a is d:
            return b
        if b is d:
            return a


def _match_mnk_rule(M, N, K, rule):
    m_max = rule.get("m_max")
    n_min = rule.get("n_min", 0)
    k_min = rule.get("k_min", 0)
    if m_max is not None and M > m_max:
        return False
    if N < n_min:
        return False
    if K < k_min:
        return False
    return True


def _select_mm_config(M, N, K):
    for rule in MM_GENERIC_CONFIG_TABLE:
        if _match_mnk_rule(M, N, K, rule):
            return rule["config"]
    return 8, 8, 8


def _select_mm_m1_config(N, K):
    for rule in MM_M1_CONFIG_TABLE:
        if N >= rule.get("n_min", 0) and K >= rule.get("k_min", 0):
            return rule["config"]
    return 64, 8


def _m1_fastpath_enabled():
    return os.getenv("FLAGGEMS_ARM_M1_FASTPATH", "0").lower() in ("1", "true", "on")


def mm(a, b):
    logging.debug("GEMS MM")
    device = a.device
    # handle non-contiguous inputs if necessary
    if a.stride(0) > 1 and a.stride(1) > 1:
        a = a.contiguous()
    if b.stride(0) > 1 and b.stride(1) > 1:
        b = b.contiguous()
    # checks constraints
    assert a.shape[1] == b.shape[0], "incompatible dimensions"
    M, K = a.shape
    _, N = b.shape
    # allocates output
    c_dtype = get_higher_dtype(a.dtype, b.dtype)
    c = torch.empty((M, N), device=device, dtype=c_dtype)
    if M == 1 and _m1_fastpath_enabled():
        m1_cfg = _select_mm_m1_config(N, K)
        if m1_cfg is not None:
            BLOCK_N, BLOCK_K = m1_cfg
            EVEN_K = K % BLOCK_K == 0
            grid = lambda META: (triton.cdiv(N, BLOCK_N),)
            mm_m1_kernel[grid](
                a,
                b,
                c,
                N,
                K,
                a.stride(1),
                b.stride(0),
                b.stride(1),
                c.stride(1),
                BLOCK_N=BLOCK_N,
                BLOCK_K=BLOCK_K,
                EVEN_K=EVEN_K,
            )
            return c

    BLOCK_M, BLOCK_N, BLOCK_K = _select_mm_config(M, N, K)
    EVEN_K = K % BLOCK_K == 0
    # launch kernel
    grid = lambda META: (
        triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),
        1,
    )
    mm_kernel[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        dot_out_dtype=tl.float32,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        GROUP_M=8,
        SPLIT_K=1,
        EVEN_K=EVEN_K,
    )
    return c


def mm_out(a, b, *, out):
    logging.debug("GEMS MM_OUT")
    if a.stride(0) > 1 and a.stride(1) > 1:
        a = a.contiguous()
    if b.stride(0) > 1 and b.stride(1) > 1:
        b = b.contiguous()

    assert a.shape[1] == b.shape[0], "incompatible dimensions"
    M, K = a.shape
    _, N = b.shape
    assert out is not None, "out tensor is required"
    assert out.shape == (M, N), "incompatible out shape"
    if M == 1 and _m1_fastpath_enabled():
        m1_cfg = _select_mm_m1_config(N, K)
        if m1_cfg is not None:
            BLOCK_N, BLOCK_K = m1_cfg
            EVEN_K = K % BLOCK_K == 0
            grid = lambda META: (triton.cdiv(N, BLOCK_N),)
            mm_m1_kernel[grid](
                a,
                b,
                out,
                N,
                K,
                a.stride(1),
                b.stride(0),
                b.stride(1),
                out.stride(1),
                BLOCK_N=BLOCK_N,
                BLOCK_K=BLOCK_K,
                EVEN_K=EVEN_K,
            )
            return out

    BLOCK_M, BLOCK_N, BLOCK_K = _select_mm_config(M, N, K)
    EVEN_K = K % BLOCK_K == 0

    grid = lambda META: (
        triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),
        1,
    )
    mm_kernel[grid](
        a,
        b,
        out,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        out.stride(0),
        out.stride(1),
        dot_out_dtype=tl.float32,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        GROUP_M=8,
        SPLIT_K=1,
        EVEN_K=EVEN_K,
    )
    return out
