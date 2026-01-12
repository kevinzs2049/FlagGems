import contextlib
import os
import time

# Configure OpenMP/MKL threads before importing torch (default to 4 for stability)
cpu_threads = int(os.environ.get("FLAGGEMS_CPU_THREADS", "4"))
os.environ.setdefault("OMP_NUM_THREADS", str(cpu_threads))
os.environ.setdefault("MKL_NUM_THREADS", str(cpu_threads))

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.streamers import BaseStreamer

import flag_gems

# 强制定为 CPU
device = "cpu"

torch.set_num_threads(cpu_threads)

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(
    "/home/kevin/.cache/modelscope/hub/models/Qwen/Qwen3-0.6B"
)
model = AutoModelForCausalLM.from_pretrained(
    "/home/kevin/.cache/modelscope/hub/models/Qwen/Qwen3-0.6B",
    torch_dtype=torch.bfloat16,
)
model.to(device).eval()


def _cooldown(seconds: int) -> None:
    print(f"\n{'=' * 40}")
    print(f"[System] Cooling down for {seconds}s to stabilize CPU state...")
    for i in range(seconds, 0, -1):
        print(f"\rWaiting... {i}s remaining", end="", flush=True)
        time.sleep(1)
    print("\n[System] Ready! Starting benchmark.")


class TokenLatencyStreamer(BaseStreamer):
    def __init__(self, prompt_tokens: int) -> None:
        super().__init__()
        self.first_token_time: float | None = None
        self.prompt_tokens = prompt_tokens
        self.total_tokens_seen = 0
        self.generated_tokens = 0

    def put(self, value) -> None:
        if isinstance(value, torch.Tensor):
            tokens = int(value.numel())
        else:
            tokens = len(value)
        self.total_tokens_seen += tokens
        generated_so_far = max(self.total_tokens_seen - self.prompt_tokens, 0)
        if generated_so_far > self.generated_tokens:
            if self.first_token_time is None:
                self.first_token_time = time.perf_counter()
            self.generated_tokens = generated_so_far

    def end(self) -> None:
        return


@pytest.mark.parametrize("enable_gems", [False,True])
@pytest.mark.parametrize("prompt", ["What is your name?"])
def test_cpu_tps_and_first_token_latency(prompt, enable_gems):
    # ==========================================
    # 1. 强制冷却/等待逻辑 (Cool Down)
    # ==========================================
    sleep_time = 30
    _cooldown(sleep_time)
    # ==========================================

    inputs = tokenizer(prompt, return_tensors="pt").to(device=device)
    input_ids = inputs["input_ids"]
    input_len = input_ids.shape[1]

    ttft_tokens = 1
    tps_tokens = 128
    repeats = 5
    beam_num = 1

    if enable_gems:
        run_context = flag_gems.use_gems()
        mode_name = "FlagGems (Optimized)"
    else:
        run_context = contextlib.nullcontext()
        mode_name = "Native PyTorch"

    print(f"[Info] Current Mode: {mode_name}")

    with run_context:
        # 2. Warmup
        print(f"[{mode_name}] Warming up...")
        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=1, num_beams=beam_num)

        print(f"[{mode_name}] Measuring prefill TPS...")

        prefill_tps_runs = []
        ttft_runs = []
        decode_tps_runs = []
        tps_runs = []
        generated_tokens = 0

        for _ in range(repeats):
            prefill_start = time.perf_counter()
            with torch.no_grad():
                model(**inputs, use_cache=True)
            prefill_end = time.perf_counter()
            prefill_duration = prefill_end - prefill_start
            prefill_tps_runs.append(
                input_len / prefill_duration if prefill_duration > 0 else 0.0
            )

            ttft_streamer = TokenLatencyStreamer(prompt_tokens=input_len)
            ttft_start = time.perf_counter()
            with torch.no_grad():
                model.generate(
                    **inputs,
                    max_new_tokens=ttft_tokens,
                    max_length=input_len + ttft_tokens,
                    num_beams=beam_num,
                    streamer=ttft_streamer,
                )
            ttft_end = time.perf_counter()

            if ttft_streamer.first_token_time is None:
                first_token_latency = ttft_end - ttft_start
            else:
                first_token_latency = ttft_streamer.first_token_time - ttft_start
            ttft_runs.append(first_token_latency)

            tps_streamer = TokenLatencyStreamer(prompt_tokens=input_len)
            tps_start = time.perf_counter()
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=tps_tokens,
                    max_length=input_len + tps_tokens,
                    num_beams=beam_num,
                    streamer=tps_streamer,
                )
            tps_end = time.perf_counter()

            total_len = outputs.shape[1]
            generated_tokens = total_len - input_len
            duration = tps_end - tps_start
            tps_runs.append(generated_tokens / duration if duration > 0 else 0.0)

            if tps_streamer.first_token_time is None:
                decode_tps_runs.append(0.0)
            else:
                decode_duration = tps_end - tps_streamer.first_token_time
                decode_tokens = max(tps_streamer.generated_tokens - 1, 0)
                decode_tps_runs.append(
                    decode_tokens / decode_duration if decode_duration > 0 else 0.0
                )

        prefill_tps = sum(prefill_tps_runs) / repeats
        first_token_latency = sum(ttft_runs) / repeats
        decode_tps = sum(decode_tps_runs) / repeats
        tps = sum(tps_runs) / repeats

        print(f"\n{'-' * 40}")
        print(f"Mode               : {mode_name}")
        print(f"Prefill TPS        : {prefill_tps:.2f} tokens/s")
        print(f"TTFT               : {first_token_latency:.4f} s")
        print(f"Decode TPS         : {decode_tps:.2f} tokens/s")
        print(f"Generated          : {generated_tokens} tokens")
        print(f"Time               : {duration:.4f} s")
        print(f"TPS                : {tps:.2f} tokens/s")
        print(f"{'-' * 40}\n")
