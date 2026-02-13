import os
import time

import pytest
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
try:
    from examples.qwen_rmsnorm_patch import enable_qwen_rmsnorm_triton_patch
except ModuleNotFoundError:
    from qwen_rmsnorm_patch import enable_qwen_rmsnorm_triton_patch  # type: ignore

device = "cpu"
MODEL_PATH = os.getenv(
    "QWEN3_MODEL_PATH", "/home/kevin/.cache/modelscope/hub/models/Qwen/Qwen3-0.6B"
)
MAX_NEW_TOKENS = int(os.getenv("QWEN3_MAX_NEW_TOKENS", "8"))
NUM_BEAMS = int(os.getenv("QWEN3_NUM_BEAMS", "1"))
PROMPTS = os.getenv("QWEN3_PROMPTS", "What is your name?").split("||")
PROFILE_TOPK = int(os.getenv("QWEN3_PROFILE_TOPK", "12"))
PROFILE_GAP_TOPK = int(os.getenv("QWEN3_PROFILE_GAP_TOPK", "20"))
LINEAR_M1_FASTPATH = os.getenv("QWEN3_LINEAR_M1_FASTPATH", "0") == "1"
LINEAR_M1_FASTPATH_SCOPE = os.getenv("QWEN3_LINEAR_M1_FASTPATH_SCOPE", "gems")
RMSNORM_TRITON_PATCH = os.getenv("QWEN3_RMSNORM_TRITON_PATCH", "0") == "1"

if RMSNORM_TRITON_PATCH:
    patched = enable_qwen_rmsnorm_triton_patch()
    print(f"[qwen3][rmsnorm-triton-patch] patched_classes={patched}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
model.to(device).eval()


def _generate_with_profile(tag, model, inputs, generate_kwargs, use_gems):
    linear_fastpath_enabled = enable_linear_m1_fastpath_for_run(
        enabled=LINEAR_M1_FASTPATH,
        scope=LINEAR_M1_FASTPATH_SCOPE,
        use_gems=use_gems,
        device=device,
    )
    start_time = time.perf_counter()
    with linear_m1_fastpath(linear_fastpath_enabled) as fastpath_stats:
        with torch.no_grad():
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU],
                profile_memory=True,
                record_shapes=False,
            ) as prof:
                output = model.generate(**inputs, **generate_kwargs)
    elapsed = time.perf_counter() - start_time
    input_tokens = int(inputs["input_ids"].shape[-1])
    total_tokens = int(output.shape[-1])
    new_tokens = max(total_tokens - input_tokens, 0)
    tok_per_s = new_tokens / elapsed if elapsed > 0 else 0.0
    print(
        f"[profile][{tag}] elapsed_s={elapsed:.3f} "
        f"input_tokens={input_tokens} new_tokens={new_tokens} "
        f"total_tokens={total_tokens} tok_per_s={tok_per_s:.3f} "
        f"linear_m1_hits={fastpath_stats['hits']}"
    )
    print(
        f"[profile][{tag}] top_cpu_ops:\n"
        f"{prof.key_averages().table(sort_by='cpu_time_total', row_limit=PROFILE_TOPK)}"
    )
    cpu_totals_ms = {}
    for evt in prof.key_averages():
        cpu_totals_ms[evt.key] = float(evt.cpu_time_total) / 1000.0
    return output, cpu_totals_ms


def _print_op_gap(ref_cpu_totals_ms, gems_cpu_totals_ms, topk):
    keys = set(ref_cpu_totals_ms.keys()) | set(gems_cpu_totals_ms.keys())
    rows = []
    for key in keys:
        ref_ms = ref_cpu_totals_ms.get(key, 0.0)
        gems_ms = gems_cpu_totals_ms.get(key, 0.0)
        delta_ms = gems_ms - ref_ms
        if ref_ms > 0.0:
            ratio = gems_ms / ref_ms
            ratio_str = f"{ratio:.2f}x"
        else:
            ratio_str = "inf" if gems_ms > 0.0 else "1.00x"
        rows.append((key, ref_ms, gems_ms, delta_ms, ratio_str))

    rows.sort(key=lambda x: x[3], reverse=True)
    print(
        f"[profile][gap] sum_ref_ms={sum(ref_cpu_totals_ms.values()):.2f} "
        f"sum_gems_ms={sum(gems_cpu_totals_ms.values()):.2f} "
        f"delta_ms={sum(gems_cpu_totals_ms.values()) - sum(ref_cpu_totals_ms.values()):.2f}"
    )
    print(
        "[profile][gap] key | ref_ms | gems_ms | delta_ms(gems-ref) | ratio(gems/ref)"
    )
    for key, ref_ms, gems_ms, delta_ms, ratio_str in rows[:topk]:
        print(
            f"[profile][gap] {key} | {ref_ms:.2f} | {gems_ms:.2f} | "
            f"{delta_ms:.2f} | {ratio_str}"
        )


@pytest.mark.parametrize(
    "prompt",
    PROMPTS,
)
def test_accuracy_Qwen(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(device=device)
    generate_kwargs = {
        "max_new_tokens": MAX_NEW_TOKENS,
        "do_sample": False,
        "num_beams": NUM_BEAMS,
    }

    ref_output, ref_cpu_totals_ms = _generate_with_profile(
        "ref", model, inputs, generate_kwargs, use_gems=False
    )

    with flag_gems.use_gems():
        res_output, gems_cpu_totals_ms = _generate_with_profile(
            "gems", model, inputs, generate_kwargs, use_gems=True
        )

    _print_op_gap(ref_cpu_totals_ms, gems_cpu_totals_ms, PROFILE_GAP_TOPK)

    generated_text = tokenizer.decode(ref_output[0], skip_special_tokens=True)
    print(generated_text)
    generated_text = tokenizer.decode(res_output[0], skip_special_tokens=True)
    print(generated_text)

    maxdiff = torch.max(torch.abs(ref_output - res_output))
    assert torch.equal(
        ref_output, res_output
    ), f"Qwen FAIL with maxdiff {maxdiff} \nREF: {ref_output}\nRES: {res_output}"
