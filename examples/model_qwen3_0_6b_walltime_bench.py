import os
import statistics
import time
from contextlib import nullcontext

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import flag_gems
try:
    from examples.qwen_linear_fastpath import (
        enable_linear_m1_fastpath_for_run,
        linear_m1_fastpath,
    )
except ModuleNotFoundError:
    from qwen_linear_fastpath import (  # type: ignore
        enable_linear_m1_fastpath_for_run,
        linear_m1_fastpath,
    )


MODEL_PATH = os.getenv(
    "QWEN3_MODEL_PATH", "/home/kevin/.cache/modelscope/hub/models/Qwen/Qwen3-0.6B"
)
PROMPT = os.getenv("QWEN3_PROMPT", "What is your name?")
MAX_NEW_TOKENS = int(os.getenv("QWEN3_MAX_NEW_TOKENS", "32"))
NUM_BEAMS = int(os.getenv("QWEN3_NUM_BEAMS", "1"))
REPEATS = int(os.getenv("QWEN3_BENCH_REPEATS", "5"))
WARMUP = int(os.getenv("QWEN3_BENCH_WARMUP", "1"))
ALTERNATE_ORDER = os.getenv("QWEN3_BENCH_ALTERNATE", "1") == "1"
LINEAR_M1_FASTPATH = os.getenv("QWEN3_LINEAR_M1_FASTPATH", "0") == "1"
LINEAR_M1_FASTPATH_SCOPE = os.getenv("QWEN3_LINEAR_M1_FASTPATH_SCOPE", "gems")
DEVICE = "cpu"


def _run_once(tag, model, tokenizer, inputs, generate_kwargs, use_gems):
    linear_fastpath_enabled = enable_linear_m1_fastpath_for_run(
        enabled=LINEAR_M1_FASTPATH,
        scope=LINEAR_M1_FASTPATH_SCOPE,
        use_gems=use_gems,
        device=DEVICE,
    )
    ctx = flag_gems.use_gems() if use_gems else nullcontext()
    start = time.perf_counter()
    with linear_m1_fastpath(linear_fastpath_enabled) as fastpath_stats:
        with ctx:
            with torch.inference_mode():
                output = model.generate(**inputs, **generate_kwargs)
    elapsed = time.perf_counter() - start
    input_tokens = int(inputs["input_ids"].shape[-1])
    total_tokens = int(output.shape[-1])
    new_tokens = max(total_tokens - input_tokens, 0)
    tok_per_s = new_tokens / elapsed if elapsed > 0 else 0.0
    print(
        f"[walltime][{tag}] elapsed_s={elapsed:.3f} input_tokens={input_tokens} "
        f"new_tokens={new_tokens} total_tokens={total_tokens} tok_per_s={tok_per_s:.3f} "
        f"linear_m1_hits={fastpath_stats['hits']}"
    )
    return elapsed, tok_per_s, output


def _percentile(values, p):
    if not values:
        return 0.0
    s = sorted(values)
    idx = (len(s) - 1) * p
    lower = int(idx)
    upper = min(lower + 1, len(s) - 1)
    frac = idx - lower
    return s[lower] * (1.0 - frac) + s[upper] * frac


def _summarize(tag, elapsed_list, tokps_list):
    elapsed_mean = statistics.mean(elapsed_list)
    tokps_mean = statistics.mean(tokps_list)
    elapsed_std = statistics.stdev(elapsed_list) if len(elapsed_list) > 1 else 0.0
    tokps_std = statistics.stdev(tokps_list) if len(tokps_list) > 1 else 0.0
    print(
        f"[walltime][summary][{tag}] runs={len(elapsed_list)} "
        f"elapsed_mean={elapsed_mean:.3f}s elapsed_std={elapsed_std:.3f}s "
        f"elapsed_p50={_percentile(elapsed_list, 0.5):.3f}s "
        f"elapsed_p90={_percentile(elapsed_list, 0.9):.3f}s "
        f"tok_per_s_mean={tokps_mean:.3f} tok_per_s_std={tokps_std:.3f}"
    )
    return elapsed_mean, tokps_mean


def main():
    print(
        f"[walltime] model={MODEL_PATH} prompt={PROMPT!r} max_new_tokens={MAX_NEW_TOKENS} "
        f"num_beams={NUM_BEAMS} repeats={REPEATS} warmup={WARMUP} "
        f"alternate_order={ALTERNATE_ORDER} linear_m1_fastpath={LINEAR_M1_FASTPATH} "
        f"linear_m1_scope={LINEAR_M1_FASTPATH_SCOPE}"
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH).to(DEVICE).eval()
    inputs = tokenizer(PROMPT, return_tensors="pt").to(device=DEVICE)
    generate_kwargs = {
        "max_new_tokens": MAX_NEW_TOKENS,
        "do_sample": False,
        "num_beams": NUM_BEAMS,
    }

    print("[walltime] warmup")
    for i in range(WARMUP):
        print(f"[walltime] warmup_round={i + 1}")
        _run_once("ref:warmup", model, tokenizer, inputs, generate_kwargs, use_gems=False)
        _run_once("gems:warmup", model, tokenizer, inputs, generate_kwargs, use_gems=True)

    elapsed = {"ref": [], "gems": []}
    tokps = {"ref": [], "gems": []}
    baseline_ref_output = None

    print("[walltime] benchmark")
    for i in range(REPEATS):
        run_id = i + 1
        order = ["ref", "gems"]
        if ALTERNATE_ORDER and i % 2 == 1:
            order = ["gems", "ref"]
        print(f"[walltime] run={run_id} order={order}")

        run_outputs = {}
        for tag in order:
            use_gems = tag == "gems"
            e, tps, out = _run_once(
                f"{tag}:run{run_id}",
                model,
                tokenizer,
                inputs,
                generate_kwargs,
                use_gems=use_gems,
            )
            elapsed[tag].append(e)
            tokps[tag].append(tps)
            run_outputs[tag] = out

        if "ref" in run_outputs and "gems" in run_outputs:
            if not torch.equal(run_outputs["ref"], run_outputs["gems"]):
                print(f"[walltime][run={run_id}] WARNING: ref and gems outputs differ")
            if baseline_ref_output is None:
                baseline_ref_output = run_outputs["ref"]
            elif not torch.equal(run_outputs["ref"], baseline_ref_output):
                print(f"[walltime][run={run_id}] WARNING: ref output differs from baseline")

    ref_elapsed_mean, ref_tokps_mean = _summarize("ref", elapsed["ref"], tokps["ref"])
    gems_elapsed_mean, gems_tokps_mean = _summarize("gems", elapsed["gems"], tokps["gems"])

    speedup = ref_tokps_mean / gems_tokps_mean if gems_tokps_mean > 0 else float("inf")
    print(
        f"[walltime][compare] ref_tok_per_s={ref_tokps_mean:.3f} "
        f"gems_tok_per_s={gems_tokps_mean:.3f} ref_over_gems={speedup:.3f}x"
    )

    if baseline_ref_output is not None:
        text = tokenizer.decode(baseline_ref_output[0], skip_special_tokens=True)
        print(f"[walltime][text] {text}")


if __name__ == "__main__":
    main()
