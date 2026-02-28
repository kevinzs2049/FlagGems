"""
Qwen3-0.6B BF16 端到端 tok/s 对比：PyTorch 原生 vs FlagGems Triton-CPU

测试配置 (CIX P1 CD8180, ARM64, 2026-02-27, performance governor 必须预先设置):
  短 prompt (8 tok, 20 new)  N_RUNS=5, 30s cooldown:
    PyTorch native  OMP=8 taskset 8 big-core : 5.61 tok/s  (baseline)
    PyTorch native  OMP=6 taskset 6 big-core : 5.86 tok/s  (同等绑核 baseline; 6核>8核因cores8,9慢)
    FlagGems Triton OMP=6 taskset 6 big-core : 6.41 tok/s  (+9% vs 同等绑核 / +14% vs 8核基线)
  128-token prompt (126 tok, 50 new):
    PyTorch native  OMP=8 taskset 8 big-core : 4.86 tok/s  (baseline)
    FlagGems Triton OMP=6 taskset 6 big-core : 4.67 tok/s  (-3.9%)  ← prefill BF16→FP32 cast

  ⚠️ 旧数据 (schedutil governor 下测量，CPU 频率未到最大) 已作废。

关键发现 — OMP 线程数不对称 (taskset 绑大核):
  Baseline (ATen)       : OMP=8 最优（多核 NEON），与 OMP=6 同等绑核下基本持平
  FlagGems (Triton-CPU) : OMP=6 最优（绑 cpu0,1,10,11,6,7 大核；
                          Triton 通过 launch grid 管理并行；
                          OMP>6 引入小核调度开销）
  ⚠️ 务必独立运行 baseline 和 FlagGems (不同进程, 不同 taskset)；
     不能在 taskset -c 0,1,10,11,6,7 下以 OMP=8 跑 baseline（8线程占6核→竞争）

算子选择说明 — decode 路径 (M=1 小张量) 的 Triton 启动开销分析:

  ✓ 有益（高计算/启动比）:
      mm/addmm/bmm/sdpa   GEMM 形状大，compute 远超 9μs 启动开销
      silu/softmax        融合减少内存往返
      rsqrt/mean          RMSNorm 组件，轻量但频率适中

  ~ 基本中性（各自 <1%，合计约 -1%）:
      sub/sub_            残差减法，调用次数少
      pow variants        RMSNorm x² 步骤
      argmax/max/min/sum  规约类，调用次数少
      log_softmax         少量调用
      embedding           prefill 主导，decode 1次
      index/index_select/gather/scatter  低频
      sort/topk/multinomial              采样时低频
      where/div variants                 低频

  ~ 轻微负面（合计约 -4~6%）:
      neg/neg_/cos/cos_/sin/sin_  M=1 Triton ~15μs vs ATen ~3μs
                                   调用次数中等（RoPE 相关）
      gelu/gelu_                   M=1 激活，类似 silu 但使用较少

  ✗ 明确有害（排除）:
      mul/mul_   weight[1024]×hidden[1,1,1024] 路由 bug 修复后仍 73μs vs ATen 3μs
                 252+ calls/token → 单独造成 -28% regression
      add/add_   residual [1,1,3584]: 56 calls × ~30μs → -8% regression
      patch_qwen3_rmsnorm()  M=1 Triton 2-pass 比 ATen 5-op 分解慢

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
# Op-group benchmark results (CIX P1 CD8180, OMP=6, taskset big-cores):
#   v1-only (12 ops):              6.51 tok/s  (reference)
#   v1 + unary(neg/cos/sin/gelu):  6.26 tok/s  (-4%)
#   v1 + sub:                      6.45 tok/s  (-1%)
#   v1 + pow:                      6.47 tok/s  (-1%)
#   v1 + redux+misc:               6.45 tok/s  (-1%)
#   v1 + expanded (this list):     6.09 tok/s  (-6%)
#   v1 + add:                      6.00 tok/s  (-8%)   ← excluded
#   v1 + mul:                      4.70 tok/s  (-28%)  ← excluded
FLAGGEMS_INCLUDE = [
    # ── GEMM / attention (high-value, always on) ──────────────────────────
    "mm", "mm_out",
    "addmm", "addmm_out",
    "bmm",
    "scaled_dot_product_attention",

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
