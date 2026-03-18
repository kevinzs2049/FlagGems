import os
import time
from collections import defaultdict
from contextlib import nullcontext

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import flag_gems


MODEL_PATH = os.getenv(
    "QWEN3_MODEL_PATH", "/home/kevin/.cache/modelscope/hub/models/Qwen/Qwen3-0.6B"
)
PROMPT = os.getenv("QWEN3_PROMPT", "What is your name?")
MAX_NEW_TOKENS = int(os.getenv("QWEN3_MAX_NEW_TOKENS", "8"))
SHAPE_TOPK = int(os.getenv("QWEN3_SHAPE_TOPK", "10"))
WARMUP_ITERS = int(os.getenv("MM_BENCH_WARMUP", "10"))
BENCH_ITERS = int(os.getenv("MM_BENCH_ITERS", "200"))
DEVICE = "cpu"


def _shape_tuple(x):
    return tuple(int(v) for v in x)


def collect_hot_shapes(model, tokenizer):
    inputs = tokenizer(PROMPT, return_tensors="pt").to(device=DEVICE)
    kwargs = {"max_new_tokens": MAX_NEW_TOKENS, "do_sample": False, "num_beams": 1}
    with torch.inference_mode():
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU],
            record_shapes=True,
            profile_memory=False,
        ) as prof:
            _ = model.generate(**inputs, **kwargs)

    mm_shapes = defaultdict(lambda: {"count": 0, "cpu_us": 0.0})
    addmm_shapes = defaultdict(lambda: {"count": 0, "cpu_us": 0.0})
    for evt in prof.events():
        name = evt.name
        inp = evt.input_shapes
        cpu_us = float(getattr(evt, "cpu_time_total", 0.0))
        if name == "aten::mm" and len(inp) >= 2:
            a = _shape_tuple(inp[0])
            b = _shape_tuple(inp[1])
            key = (a, b)
            mm_shapes[key]["count"] += 1
            mm_shapes[key]["cpu_us"] += cpu_us
        elif name == "aten::addmm" and len(inp) >= 3:
            bias = _shape_tuple(inp[0])
            a = _shape_tuple(inp[1])
            b = _shape_tuple(inp[2])
            key = (bias, a, b)
            addmm_shapes[key]["count"] += 1
            addmm_shapes[key]["cpu_us"] += cpu_us

    mm_rank = sorted(mm_shapes.items(), key=lambda x: x[1]["cpu_us"], reverse=True)
    addmm_rank = sorted(addmm_shapes.items(), key=lambda x: x[1]["cpu_us"], reverse=True)
    return mm_rank[:SHAPE_TOPK], addmm_rank[:SHAPE_TOPK]


def _bench_loop(fn, warmup_iters=WARMUP_ITERS, bench_iters=BENCH_ITERS):
    for _ in range(warmup_iters):
        fn()
    t0 = time.perf_counter()
    for _ in range(bench_iters):
        fn()
    elapsed = time.perf_counter() - t0
    return elapsed * 1000.0 / bench_iters


def bench_mm_shape(shape_pair, use_gems):
    (m, k), (k2, n) = shape_pair
    assert k == k2
    a = torch.randn((m, k), device=DEVICE, dtype=torch.float32)
    b = torch.randn((k, n), device=DEVICE, dtype=torch.float32)
    ctx = flag_gems.use_gems() if use_gems else nullcontext()
    with ctx:
        return _bench_loop(lambda: torch.mm(a, b))


def bench_addmm_shape(shape_triplet, use_gems):
    (m1, n1), (m2, k), (k2, n2) = shape_triplet
    assert m1 == m2 and n1 == n2 and k == k2
    bias = torch.randn((m1, n1), device=DEVICE, dtype=torch.float32)
    a = torch.randn((m2, k), device=DEVICE, dtype=torch.float32)
    b = torch.randn((k2, n2), device=DEVICE, dtype=torch.float32)
    ctx = flag_gems.use_gems() if use_gems else nullcontext()
    with ctx:
        return _bench_loop(lambda: torch.addmm(bias, a, b))


def _print_mm_results(mm_rank):
    print("\n[mm-bench] aten::mm hot shapes from model.generate")
    print("[mm-bench] shape(A) x shape(B) | profile_calls | profile_cpu_ms | ref_ms | gems_ms | ref/gems")
    for (shape_pair, stat) in mm_rank:
        ref_ms = bench_mm_shape(shape_pair, use_gems=False)
        gems_ms = bench_mm_shape(shape_pair, use_gems=True)
        speed = ref_ms / gems_ms if gems_ms > 0 else float("inf")
        a, b = shape_pair
        print(
            f"[mm-bench] {a} x {b} | {stat['count']} | {stat['cpu_us'] / 1000.0:.3f} | "
            f"{ref_ms:.3f} | {gems_ms:.3f} | {speed:.3f}x"
        )


def _print_addmm_results(addmm_rank):
    print("\n[mm-bench] aten::addmm hot shapes from model.generate")
    print(
        "[mm-bench] shape(bias),shape(A),shape(B) | profile_calls | profile_cpu_ms | "
        "ref_ms | gems_ms | ref/gems"
    )
    for (shape_triplet, stat) in addmm_rank:
        ref_ms = bench_addmm_shape(shape_triplet, use_gems=False)
        gems_ms = bench_addmm_shape(shape_triplet, use_gems=True)
        speed = ref_ms / gems_ms if gems_ms > 0 else float("inf")
        bias, a, b = shape_triplet
        print(
            f"[mm-bench] {bias},{a},{b} | {stat['count']} | {stat['cpu_us'] / 1000.0:.3f} | "
            f"{ref_ms:.3f} | {gems_ms:.3f} | {speed:.3f}x"
        )


def main():
    os.environ.setdefault("GEMS_VENDOR", "arm")
    print(
        f"[mm-bench] model={MODEL_PATH} prompt={PROMPT!r} "
        f"max_new_tokens={MAX_NEW_TOKENS} topk={SHAPE_TOPK} "
        f"warmup_iters={WARMUP_ITERS} bench_iters={BENCH_ITERS}"
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH).to(DEVICE).eval()
    mm_rank, addmm_rank = collect_hot_shapes(model, tokenizer)
    _print_mm_results(mm_rank)
    _print_addmm_results(addmm_rank)


if __name__ == "__main__":
    main()
