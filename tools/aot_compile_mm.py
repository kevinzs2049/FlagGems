#!/usr/bin/env python3
"""
AOT compile mm_m1_kernel variants for ARM64 decode inference.

Pre-compiles all commonly used specializations of mm_m1_kernel (M=1 decode GEMV)
with the same constexpr and divisibility optimizations as JIT auto-specialization,
so they can be loaded at runtime without JIT overhead.

Usage:
    python tools/aot_compile_mm.py --model qwen3-0.6b
    python tools/aot_compile_mm.py --model qwen2-7b
    python tools/aot_compile_mm.py --all-models

    # Compile to a specific directory:
    FLAGGEMS_AOT_DIR=/path/to/cache python tools/aot_compile_mm.py

Compiled kernels are loaded automatically when FLAGGEMS_USE_AOT=1.
"""

import argparse
import logging
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# Model shape definitions: (N, K) pairs from linear layers during M=1 decode
MODEL_SHAPES = {
    "qwen3-0.6b": [
        (1024, 1024),   # q_proj, o_proj
        (512, 1024),    # k_proj, v_proj
        (2816, 1024),   # gate_proj, up_proj
        (1024, 2816),   # down_proj
        (151936, 1024), # lm_head
    ],
    "qwen2-7b": [
        (3584, 3584),    # q_proj, o_proj
        (512, 3584),     # k_proj, v_proj
        (18944, 3584),   # gate_proj, up_proj
        (3584, 18944),   # down_proj
        (152064, 3584),  # lm_head
    ],
    "qwen3-1.7b": [
        (2048, 2048),   # q_proj, o_proj
        (1024, 2048),   # k_proj, v_proj
        (5632, 2048),   # gate_proj, up_proj
        (2048, 5632),   # down_proj
        (151936, 2048), # lm_head
    ],
}


def compile_model_kernels(model_name: str, cache_dir: str = None, dtypes=None):
    """Compile all AOT kernel variants for a specific model."""
    from flag_gems.runtime.aot_loader import AOTKernelCache
    from flag_gems.runtime.backend._arm.ops.mm import (
        mm_m1_kernel, _select_mm_m1_config,
    )

    if model_name not in MODEL_SHAPES:
        logger.error(f"Unknown model: {model_name}. Available: {list(MODEL_SHAPES.keys())}")
        return

    shapes = MODEL_SHAPES[model_name]
    if dtypes is None:
        dtypes = ["bf16"]

    cache = AOTKernelCache(cache_dir)
    total_compiled = 0
    total_skipped = 0
    t_start = time.time()

    for dtype in dtypes:
        logger.info(f"\n=== Compiling {model_name} decode kernels (dtype={dtype}) ===")

        # mm_m1_kernel runtime (non-constexpr) args for this dtype
        ptr_type = f"*{dtype}"
        out_ptr_type = "*fp32" if dtype == "bf16" else ptr_type
        signature = {
            "A": ptr_type,
            "B": ptr_type,
            "C": out_ptr_type,
            "N": "i32",
            "K": "i32",
            "stride_bk": "i32",
        }

        seen_variants = set()

        for N, K in shapes:
            m1_cfg = _select_mm_m1_config(N, K)
            if m1_cfg is None:
                logger.info(f"  N={N}, K={K}: no M1 config, skipping")
                continue

            BLOCK_N, BLOCK_K = m1_cfg
            EVEN_K = (K % BLOCK_K == 0)

            # Constexprs: block sizes + contiguous strides (matching JIT)
            constexprs = {
                "BLOCK_N": BLOCK_N,
                "BLOCK_K": BLOCK_K,
                "EVEN_K": EVEN_K,
                "stride_ak": 1,
                "stride_bn": 1,
                "stride_cn": 1,
            }

            key = (dtype, tuple(sorted(constexprs.items())))
            if key in seen_variants:
                total_skipped += 1
                continue
            seen_variants.add(key)

            # Divisibility-16 attrs (matching JIT auto-specialization)
            # Arg indices after constexpr removal:
            # A(0), B(1), C(2), N(3), K(4), stride_bk(6)
            # stride_ak(5), stride_bn(7), stride_cn(8) are constexprs
            attrs = {
                (0,): [["tt.divisibility", 16]],  # A pointer
                (1,): [["tt.divisibility", 16]],  # B pointer
                (2,): [["tt.divisibility", 16]],  # C pointer
                (3,): [["tt.divisibility", 16]],  # N
                (4,): [["tt.divisibility", 16]],  # K
                (6,): [["tt.divisibility", 16]],  # stride_bk
            }

            logger.info(f"  N={N}, K={K}: BLOCK_N={BLOCK_N}, BLOCK_K={BLOCK_K}, EVEN_K={EVEN_K}")
            result = cache.compile_kernel(
                mm_m1_kernel, constexprs, signature, attrs=attrs,
            )
            if result:
                total_compiled += 1
            else:
                logger.warning(f"  FAILED: N={N}, K={K}")

            # Also compile EVEN_K=True if K is divisible (for padded K)
            if not EVEN_K:
                constexprs_even = dict(constexprs)
                constexprs_even["EVEN_K"] = True
                key2 = (dtype, tuple(sorted(constexprs_even.items())))
                if key2 not in seen_variants:
                    seen_variants.add(key2)
                    result = cache.compile_kernel(
                        mm_m1_kernel, constexprs_even, signature, attrs=attrs,
                    )
                    if result:
                        total_compiled += 1

    elapsed = time.time() - t_start
    logger.info(f"\nDone: {total_compiled} compiled, {total_skipped} dedup, {elapsed:.1f}s")
    cached = cache.list_cached()
    for kernel_name, variants in cached.items():
        logger.info(f"  {kernel_name}: {len(variants)} variants")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", "-m", type=str, default=None,
                        help=f"Model name. Available: {list(MODEL_SHAPES.keys())}")
    parser.add_argument("--all-models", action="store_true",
                        help="Compile for all known models")
    parser.add_argument("--dtype", type=str, nargs="+", default=["bf16"])
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--list", action="store_true", help="List cached kernels")
    args = parser.parse_args()

    if args.list:
        from flag_gems.runtime.aot_loader import AOTKernelCache
        cache = AOTKernelCache(args.cache_dir)
        for kernel_name, variants in cache.list_cached().items():
            print(f"\n{kernel_name}:")
            for key, info in variants.items():
                print(f"  [{key}] {info['constexprs']}")
        return

    if args.all_models:
        for name in MODEL_SHAPES:
            compile_model_kernels(name, args.cache_dir, args.dtype)
    elif args.model:
        compile_model_kernels(args.model, args.cache_dir, args.dtype)
    else:
        logger.info("No model specified, compiling for qwen3-0.6b")
        compile_model_kernels("qwen3-0.6b", args.cache_dir, args.dtype)


if __name__ == "__main__":
    main()
