"""
Qwen3-0.6B BF16 端到端 tok/s 对比：PyTorch 原生 vs FlagGems Triton-CPU

⚠️ 测试结论 (CIX P1 CD8180, ARM64, performance governor, 2026-03-04):
  短 prompt (8 tok, 20 new), N_RUNS=3:
    PyTorch native  OMP=6 taskset 6 big-core : 6.43 tok/s  (baseline)
    FlagGems Triton OMP=6 taskset 6 big-core : 4.03 tok/s  (-37% regression)

  ❌ FlagGems 对 Qwen3-0.6B 无益。根本原因:
     CIX P1 CD8180 原生支持 ARM BF16 BLAS (BFDOT/BFMMLA via NEON)；
     ATen 已使用原生 BF16，BF16 mm 446μs < FP32 645μs（K=1024,N=3072）。
     Triton 对所有 Qwen3-0.6B decode 形状慢 1.4-1.6x；加上 dispatch 开销 → -37%。
     FlagGems 仅对更大模型 (Qwen2-7B BF16: +2.56x) 有益，因那些形状 ATen 走 FP32 路径。

  ❌ 旧数据 (schedutil governor / wrong methodology) 已作废:
     "5.86 ATen / 6.41 FlagGems +9.4%"  — 使用 schedutil 时测量，不代表实际性能。

⚠️ only_enable() 累积 bug:
  重复调用 only_enable(include=[new_ops]) 会「追加」，不会清除旧注册。
  空列表调用静默失败 (warning only)，不取消注册。隔离测试必须用独立进程。

算子选择说明 (保留此列表以供参考，但对 Qwen3-0.6B 均为负面):
  ✗ 所有算子组合均导致 regression:
      mm/addmm/bmm        -27% (ATen BF16 BLAS 已最优)
      silu/gelu/softmax   -11% (overhead > fusion benefit)
      cos/sin/neg         -13% (M=1 Triton ~15μs vs ATen ~3μs)
      rsqrt/mean/redux    -33% (低频，累积开销)
      sdpa                -16% (decode Q_CTX=1，28 blocks → 明确有害，已从列表移除)
      mul/mul_            -28% (252+ calls/token, 73μs vs ATen 3μs)
      add/add_            -8%  (56 calls/token, ~30μs each)
      patch_qwen3_rmsnorm M=1 Triton 2-pass 比 ATen 5-op 分解慢

注意: flag_gems.enable() 全量注册与 transformers 生成循环存在兼容问题
(arange/copy_ 等工具类 op 会干扰 cache_position 处理)，
使用 only_enable() 只注册计算类 kernel。

用法:
  taskset -c 0,1,10,11,6,7 python model_qwen3_0_6b_e2e_bench.py
  taskset -c 0,1,10,11,6,7 python model_qwen3_0_6b_e2e_bench.py --baseline-only
"""

import argparse
import os
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = os.getenv(
    "QWEN3_MODEL_PATH",
    "/home/kevin/.cache/modelscope/hub/models/Qwen/Qwen3-0___6B",
)
PROMPT = os.getenv("QWEN3_PROMPT", "请简要介绍人工智能的发展历史。")
N_TOKENS = int(os.getenv("N_TOKENS", "20"))
N_RUNS = int(os.getenv("N_RUNS", "3"))

# FlagGems ops to enable via only_enable().
# All operators listed here have _arm/ops/ Triton-CPU implementations.
#
# ⚠️ NOTE (2026-03-04): ALL ops below cause regression for Qwen3-0.6B on CIX P1 CD8180.
# ATen already uses native ARM BF16 BLAS (BFDOT/BFMMLA); Triton is 1.4-1.6x slower
# for these small decode shapes. Full list gives 4.03 tok/s vs 6.43 ATen (-37%).
# This list may still be useful for other models/hardware where ATen lacks BF16 BLAS.
#
# ❌ OLD data (schedutil governor) was wrong: "6.51 ref / 6.09 expanded (-6%)" invalidated.
#
# Confirmed excluded (separate process isolation tests):
#   mul/mul_: 252+ calls/token × 73μs each → -28% e2e
#   add/add_: 56 calls/token × ~30μs each  → -8% e2e
#   sdpa: decode Q_CTX=1, 28 blocks        → -16% e2e
FLAGGEMS_INCLUDE = [
    # ── GEMM (high-value, always on) ──────────────────────────────────────
    "mm", "mm_out",
    "addmm", "addmm_out",
    "bmm",
    # NOTE: scaled_dot_product_attention excluded — isolation test shows -16%
    # regression (genuine Triton SDPA kernel overhead vs ATen for decode Q_CTX=1,
    # 28 blocks only).  Autotune key change doesn't help; removal is the fix.

    # ── Activations ───────────────────────────────────────────────────────
    "silu", "silu_",
    "gelu", "gelu_",
    "softmax",
    "log_softmax",

    # ── Normalization components ───────────────────────────────────────────
    "rsqrt", "rsqrt_",
    "mean", "mean_dim",

    # ── Elementwise unary (~-4% combined due to M=1 overhead) ─────────────
    "neg", "neg_",
    "cos", "cos_",
    "sin", "sin_",

    # ── sub: neutral (-1%) ────────────────────────────────────────────────
    "sub", "sub_",

    # ── Power: neutral (-1%) ──────────────────────────────────────────────
    "pow_scalar",
    "pow_tensor_scalar", "pow_tensor_scalar_",
    "pow_tensor_tensor", "pow_tensor_tensor_",

    # ── Division: low-frequency, neutral ──────────────────────────────────
    "true_divide", "true_divide_",
    "floor_divide", "floor_divide_",
    "remainder", "remainder_",
    "div_mode", "div_mode_",

    # ── Reduction: neutral ────────────────────────────────────────────────
    "sum", "sum_dim",
    "max", "max_dim",
    "min",
    "argmax",

    # ── Indexing / gather: low-frequency, neutral ─────────────────────────
    "embedding",
    "index",
    "index_select",
    "gather",
    "scatter", "scatter_",
    "masked_fill", "masked_fill_",

    # ── Sorting / sampling: low-frequency, neutral ────────────────────────
    "sort", "sort_stable",
    "topk",
    "multinomial",

    # ── Misc: neutral ─────────────────────────────────────────────────────
    "where_self_out",

    # ── EXCLUDED (decode performance regression) ──────────────────────────
    # "mul", "mul_",   weight[1024]×hidden[1,1,1024]: 73μs vs ATen 3μs
    #                  252+ calls/token → -28% e2e regression
    # "add", "add_",   residual [1,1,3584]: 56 calls/token → -8% regression
]


def bench(model, tokenizer, input_ids, attention_mask, label):
    n_prompt = input_ids.shape[1]

    def gen(n):
        with torch.no_grad():
            return model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=n,
                do_sample=False,
                temperature=None,
                top_p=None,
            )

    print(f"\n[{label}] Warmup 4 tokens...", flush=True)
    gen(4)
    print("  Done", flush=True)

    tps_list = []
    for i in range(N_RUNS):
        t0 = time.perf_counter()
        out = gen(N_TOKENS)
        elapsed = time.perf_counter() - t0
        tps = (out.shape[1] - n_prompt) / elapsed
        tps_list.append(tps)
        print(f"  run {i+1}: {tps:.2f} tok/s", flush=True)

    med = sorted(tps_list)[N_RUNS // 2]
    text = tokenizer.decode(out[0][n_prompt:], skip_special_tokens=True)[:60]
    print(f"  median: {med:.2f} tok/s  |  生成: {text}")
    return med


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--skip-baseline", action="store_true",
                        help="Skip baseline run; only run FlagGems (use with taskset for big-core pinning)")
    parser.add_argument("--flaggems-src", default="/home/kevin/FlagGems-rebase/FlagGems/src")
    # OMP tuning: Baseline (ATen) peaks at OMP=6-8; FlagGems (Triton) peaks at OMP=1-2.
    # Triton-CPU manages its own kernel parallelism via the launch grid; extra OMP threads
    # cause scheduling overhead without benefiting Triton kernels.
    parser.add_argument("--baseline-omp", type=int, default=8,
                        help="OMP threads for PyTorch baseline (default=8, optimal ~6-8)")
    parser.add_argument("--flaggems-omp", type=int, default=6,
                        help="OMP threads for FlagGems Triton (default=6, optimal with taskset big-cores)")
    args = parser.parse_args()

    print(f"MODEL={MODEL_PATH}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    input_ids = tokenizer(PROMPT, return_tensors="pt")["input_ids"]
    attention_mask = torch.ones_like(input_ids)
    print(f"Prompt: {input_ids.shape[1]} tokens, max_new_tokens={N_TOKENS}")

    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True
    )
    model.eval()
    print(f"Model loaded in {time.perf_counter()-t0:.1f}s")

    # --- Baseline (ATen, optimal OMP=6-8) ---
    baseline_tps = None
    if not args.skip_baseline:
        import os as _os
        affinity = _os.sched_getaffinity(0)
        if len(affinity) < args.baseline_omp:
            print(f"\n⚠️  WARNING: CPU affinity={sorted(affinity)} has only {len(affinity)} cores "
                  f"but baseline-omp={args.baseline_omp}. "
                  f"Run baseline WITHOUT taskset for accurate results.")
        torch.set_num_threads(args.baseline_omp)
        print(f"\n[Baseline] OMP={args.baseline_omp}  affinity={sorted(affinity)}")
        baseline_tps = bench(model, tokenizer, input_ids, attention_mask, "PyTorch native")

    if args.baseline_only:
        print(f"\n{'='*55}")
        print(f"  PyTorch native  OMP={args.baseline_omp}  {baseline_tps:.2f} tok/s")
        print(f"{'='*55}")
        return

    # --- FlagGems (Triton, optimal OMP=1-2) ---
    # Triton-CPU kernels run their own parallelism via launch grid; extra OMP threads
    # add scheduling overhead without improving kernel throughput for decode M=1.
    torch.set_num_threads(args.flaggems_omp)
    if args.flaggems_src not in sys.path:
        sys.path.insert(0, args.flaggems_src)
    import flag_gems
    flag_gems.only_enable(include=FLAGGEMS_INCLUDE)
    print(f"\n[FlagGems] OMP={args.flaggems_omp}  only_enable: {FLAGGEMS_INCLUDE}")

    # NOTE: patch_qwen3_rmsnorm() is available but NOT applied here.
    # For decode (M=1) the 2-pass Triton kernel is slower than ATen decomposition.
    # For prefill (M≥32) it can be beneficial:
    #   from flag_gems.runtime.backend._arm.fused import patch_qwen3_rmsnorm
    #   patch_qwen3_rmsnorm()

    fg_tps = bench(model, tokenizer, input_ids, attention_mask, "FlagGems Triton")

    print(f"\n{'='*55}")
    print(f"  N_TOKENS={N_TOKENS}  N_RUNS={N_RUNS}")
    if baseline_tps is not None:
        print(f"  PyTorch native (OMP={args.baseline_omp}): {baseline_tps:.2f} tok/s")
        print(f"  FlagGems Triton (OMP={args.flaggems_omp}): {fg_tps:.2f} tok/s  ({fg_tps/baseline_tps:.2f}x)")
    else:
        print(f"  FlagGems Triton (OMP={args.flaggems_omp}): {fg_tps:.2f} tok/s  (baseline: run separately without taskset)")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
