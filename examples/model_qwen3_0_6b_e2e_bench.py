"""
Qwen3-0.6B BF16 端到端 tok/s 对比：PyTorch 原生 vs FlagGems Triton-CPU

测试配置 (CIX P1 CD8180, ARM64, OMP=8, 2026-02-26):
  PyTorch native : 5.24 tok/s
  FlagGems Triton: 5.88 tok/s  (+12%)
  FlagGems +mul  : ~+3%（在 v1 基础上）

算子选择说明（decode 路径，M=1 小张量）:
  有益 (√):  mm/addmm/bmm —— GEMM 形状大，Triton 收益 > 启动开销
             silu/softmax/sdpa —— 融合节省多次内存往返
             rsqrt/mean —— RMSNorm 组件，轻量

  不启用 (×): mul/mul_  —— weight[1024] × hidden[1,1,1024] 广播走 @pointwise_dynamic
                           ATen ~2μs(NEON) vs FlagGems ~20μs; 392 calls/token = -19%
              add/add_  —— decode 残差 56 calls × overhead > 计算节省
              patch_qwen3_rmsnorm() —— M=1 Triton 2-pass 比 ATen 5-op 分解慢
                                      prefill(M≥32) 时可考虑开启

注意: flag_gems.enable() 全量注册与 transformers 生成循环存在兼容问题
(index/arange 等工具类 op 会破坏 cache_position 处理)，
建议使用 only_enable() 只注册计算类 kernel。

用法:
  # 仅基线
  OMP_NUM_THREADS=8 python model_qwen3_0_6b_e2e_bench.py --baseline-only

  # 基线 + FlagGems 对比
  OMP_NUM_THREADS=8 python model_qwen3_0_6b_e2e_bench.py
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

# FlagGems ops to enable (only compute-critical kernels)
FLAGGEMS_INCLUDE = [
    "mm", "mm_out",
    "addmm", "addmm_out",
    "bmm",
    "silu",
    "softmax",
    "scaled_dot_product_attention",
    "rsqrt", "rsqrt_",
    "mean", "mean_dim",
    # NOT included for decode (all three hurt decode performance due to Triton overhead):
    # "mul", "mul_",     # weight[1024] × hidden[1,1,1024] → broadcast → @pointwise_dynamic
    #                    # ATen ~2μs (NEON-optimized) vs FlagGems ~20μs; 392 calls/token = -19%
    # "add", "add_",     # residual [1,1,1024]: 56 calls × overhead > compute savings
    # patch_qwen3_rmsnorm()  # M=1 2-pass Triton slower than ATen 5-op decomposition
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
    parser.add_argument("--flaggems-src", default="/home/kevin/FlagGems-rebase/FlagGems/src")
    # OMP tuning: Baseline (ATen) peaks at OMP=6-8; FlagGems (Triton) peaks at OMP=1-2.
    # Triton-CPU manages its own kernel parallelism via the launch grid; extra OMP threads
    # cause scheduling overhead without benefiting Triton kernels.
    parser.add_argument("--baseline-omp", type=int, default=8,
                        help="OMP threads for PyTorch baseline (default=8, optimal ~6-8)")
    parser.add_argument("--flaggems-omp", type=int, default=1,
                        help="OMP threads for FlagGems Triton (default=1, optimal ~1-2)")
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
    torch.set_num_threads(args.baseline_omp)
    print(f"\n[Baseline] OMP={args.baseline_omp}")
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
    print(f"  PyTorch native (OMP={args.baseline_omp}): {baseline_tps:.2f} tok/s")
    print(f"  FlagGems Triton (OMP={args.flaggems_omp}): {fg_tps:.2f} tok/s  ({fg_tps/baseline_tps:.2f}x)")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
