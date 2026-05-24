import os
import time
from contextlib import nullcontext

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import flag_gems


MODEL_PATH = os.getenv(
    "QWEN3_MODEL_PATH", "/home/kevin/.cache/modelscope/hub/models/Qwen/Qwen3-0.6B"
)
PROMPT = os.getenv("QWEN3_PROMPT", "What is your name?")
MAX_NEW_TOKENS = int(os.getenv("QWEN3_MAX_NEW_TOKENS", "32"))
NUM_BEAMS = int(os.getenv("QWEN3_NUM_BEAMS", "1"))
PROFILE_TOPK = int(os.getenv("QWEN3_PROFILE_TOPK", "12"))
MODES = [m.strip() for m in os.getenv("QWEN3_CACHE_MODES", "default,dynamic,static,no_cache").split(",") if m.strip()]
INCLUDE_REF = os.getenv("QWEN3_COMPARE_INCLUDE_REF", "0") == "1"
DEVICE = "cpu"


def _cache_kwargs(mode):
    if mode == "default":
        return {}
    if mode == "dynamic":
        return {"cache_implementation": "dynamic"}
    if mode == "static":
        return {"cache_implementation": "static"}
    if mode == "no_cache":
        return {"use_cache": False}
    raise ValueError(f"Unsupported mode: {mode}")


def _run_generate(tag, model, tokenizer, use_gems, mode):
    inputs = tokenizer(PROMPT, return_tensors="pt").to(device=DEVICE)
    generate_kwargs = {
        "max_new_tokens": MAX_NEW_TOKENS,
        "do_sample": False,
        "num_beams": NUM_BEAMS,
    }
    generate_kwargs.update(_cache_kwargs(mode))

    ctx = flag_gems.use_gems() if use_gems else nullcontext()
    start = time.perf_counter()
    with ctx:
        with torch.inference_mode():
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU],
                profile_memory=True,
                record_shapes=False,
            ) as prof:
                output = model.generate(**inputs, **generate_kwargs)
    elapsed = time.perf_counter() - start

    input_tokens = int(inputs["input_ids"].shape[-1])
    total_tokens = int(output.shape[-1])
    new_tokens = max(total_tokens - input_tokens, 0)
    tok_per_s = new_tokens / elapsed if elapsed > 0 else 0.0

    cpu_totals = {}
    for evt in prof.key_averages():
        cpu_totals[evt.key] = float(evt.cpu_time_total)
    cat_ms = cpu_totals.get("aten::cat", 0.0) / 1000.0
    copy_ms = cpu_totals.get("aten::copy_", 0.0) / 1000.0
    clone_ms = cpu_totals.get("aten::clone", 0.0) / 1000.0

    print(
        f"[cache-compare][{tag}:{mode}] elapsed_s={elapsed:.3f} "
        f"new_tokens={new_tokens} tok_per_s={tok_per_s:.3f} "
        f"cat_ms={cat_ms:.2f} copy_ms={copy_ms:.2f} clone_ms={clone_ms:.2f}"
    )
    print(
        f"[cache-compare][{tag}:{mode}] top_cpu_ops:\n"
        f"{prof.key_averages().table(sort_by='cpu_time_total', row_limit=PROFILE_TOPK)}"
    )
    return output, tok_per_s, cat_ms, copy_ms, clone_ms


def main():
    print(
        f"[cache-compare] model={MODEL_PATH} prompt={PROMPT!r} "
        f"max_new_tokens={MAX_NEW_TOKENS} num_beams={NUM_BEAMS} modes={MODES}"
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH).to(DEVICE).eval()

    runs = [("gems", True)]
    if INCLUDE_REF:
        runs.insert(0, ("ref", False))

    for tag, use_gems in runs:
        print(f"\n[cache-compare] start run={tag}")
        baseline_output = None
        baseline_mode = None
        for mode in MODES:
            try:
                output, tok_per_s, cat_ms, copy_ms, clone_ms = _run_generate(
                    tag, model, tokenizer, use_gems, mode
                )
            except Exception as exc:
                print(f"[cache-compare][{tag}:{mode}] ERROR: {exc}")
                continue

            if baseline_output is None:
                baseline_output = output
                baseline_mode = mode
            else:
                if not torch.equal(output, baseline_output):
                    print(
                        f"[cache-compare][{tag}:{mode}] WARNING: output differs "
                        f"from baseline mode {baseline_mode}"
                    )

            text = tokenizer.decode(output[0], skip_special_tokens=True)
            print(f"[cache-compare][{tag}:{mode}] text={text}")


if __name__ == "__main__":
    main()
