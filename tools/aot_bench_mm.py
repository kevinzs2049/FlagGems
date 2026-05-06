#!/usr/bin/env python3
"""
Benchmark JIT vs AOT dispatch overhead for mm_m1_kernel.

Usage:
    # First compile AOT kernels:
    python tools/aot_compile_mm.py --model qwen3-0.6b

    # Then benchmark:
    FLAGGEMS_USE_AOT=1 python tools/aot_bench_mm.py

    # Benchmark specific shapes:
    python tools/aot_bench_mm.py --shapes "1024,1024" "2816,1024"
"""

import argparse
import logging
import os
import sys
import time

import numpy as np
import torch
import triton

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--shapes", nargs="+", type=str,
                        default=["512,1024", "1024,1024", "2816,1024", "1024,2816"],
                        help="N,K shapes to benchmark")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--n-warmup", type=int, default=30)
    parser.add_argument("--n-iter", type=int, default=500)
    args = parser.parse_args()

    dtype = getattr(torch, args.dtype)
    triton_dtype = "bf16" if dtype == torch.bfloat16 else "fp32"
    out_ptr_type = "*fp32" if triton_dtype == "bf16" else f"*{triton_dtype}"

    from flag_gems.runtime.backend._arm.ops.mm import mm_m1_kernel, _select_mm_m1_config
    from flag_gems.runtime.aot_loader import AOTKernelCache

    cache = AOTKernelCache()

    print(f"{'N':>6s} {'K':>6s} {'grid':>5s} | {'JIT(us)':>8s} {'AOT(us)':>8s} | {'ratio':>7s} {'saved':>9s} | ok")
    print("-" * 75)

    for shape_str in args.shapes:
        N, K = map(int, shape_str.split(","))
        m1_cfg = _select_mm_m1_config(N, K)
        if m1_cfg is None:
            print(f"{N:6d} {K:6d}       | no M1 config")
            continue

        BLOCK_N, BLOCK_K = m1_cfg
        EVEN_K = K % BLOCK_K == 0
        grid_size = (N + BLOCK_N - 1) // BLOCK_N

        constexprs = {
            "stride_ak": 1, "stride_bn": 1, "stride_cn": 1,
            "BLOCK_N": BLOCK_N, "BLOCK_K": BLOCK_K, "EVEN_K": EVEN_K,
        }
        aot = cache.load_kernel("mm_m1_kernel", constexprs)

        a = torch.randn(1, K, dtype=dtype)
        b = torch.randn(K, N, dtype=dtype)
        c_jit = torch.empty(1, N, dtype=torch.float32 if dtype == torch.bfloat16 else dtype)
        c_aot = torch.empty_like(c_jit)

        grid = lambda META: (triton.cdiv(N, BLOCK_N),)

        # Verify correctness
        mm_m1_kernel[grid](a, b, c_jit, N, K,
            a.stride(1), b.stride(0), b.stride(1), c_jit.stride(1),
            BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, EVEN_K=EVEN_K)

        if aot:
            aot(grid_size, 1, 1, a, b, c_aot, N, K, b.stride(0))
            ok = torch.allclose(c_jit, c_aot, atol=1e-3, rtol=1e-3)
        else:
            ok = "N/A"

        # Warmup
        for _ in range(args.n_warmup):
            mm_m1_kernel[grid](a, b, c_jit, N, K,
                a.stride(1), b.stride(0), b.stride(1), c_jit.stride(1),
                BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, EVEN_K=EVEN_K)
        if aot:
            for _ in range(args.n_warmup):
                aot(grid_size, 1, 1, a, b, c_aot, N, K, b.stride(0))

        # Benchmark JIT
        jit_t = []
        for _ in range(args.n_iter):
            t0 = time.perf_counter_ns()
            mm_m1_kernel[grid](a, b, c_jit, N, K,
                a.stride(1), b.stride(0), b.stride(1), c_jit.stride(1),
                BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, EVEN_K=EVEN_K)
            jit_t.append(time.perf_counter_ns() - t0)

        jm = np.median(jit_t) / 1e3

        if aot:
            aot_t = []
            for _ in range(args.n_iter):
                t0 = time.perf_counter_ns()
                aot(grid_size, 1, 1, a, b, c_aot, N, K, b.stride(0))
                aot_t.append(time.perf_counter_ns() - t0)
            am = np.median(aot_t) / 1e3
            print(f"{N:6d} {K:6d} {grid_size:5d} | {jm:8.1f} {am:8.1f} | {jm/am:7.2f}x {jm-am:+9.1f}us | {ok}")
        else:
            print(f"{N:6d} {K:6d} {grid_size:5d} | {jm:8.1f}     N/A |     N/A       N/A | (no AOT)")


if __name__ == "__main__":
    main()
