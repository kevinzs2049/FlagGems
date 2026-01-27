import contextlib
import os
import time

# Configure OpenMP/MKL threads before importing torch
cpu_threads = int(os.environ.get("FLAGGEMS_CPU_THREADS", "12"))
os.environ.setdefault("OMP_NUM_THREADS", str(cpu_threads))
os.environ.setdefault("MKL_NUM_THREADS", str(cpu_threads))

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.streamers import BaseStreamer

import flag_gems

device = "cpu"
torch.set_num_threads(cpu_threads)

if not hasattr(torch.cpu, "device"):
    @contextlib.contextmanager
    def cpu_device_dummy(device_index=None):
        # 这是一个空的上下文管理器，什么都不做，只是为了骗过 FlagGems 的调用
        yield

    # 强行给 torch.cpu 塞入这个方法
    torch.cpu.device = cpu_device_dummy
    print("[Hack] torch.cpu.device has been monkey-patched for FlagGems compatibility.")



model_path = os.environ.get(
    "QWEN3_MODEL_PATH",
    "/home/kevin/.cache/modelscope/hub/models/Qwen/Qwen3-0.6B",
)

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

dtype_name = os.environ.get("QWEN3_DTYPE", "fp32").lower()
if dtype_name in ("bf16", "bfloat16"):
    model_dtype = torch.bfloat16
elif dtype_name in ("fp32", "float32"):
    model_dtype = torch.float32
else:
    raise ValueError(f"Unsupported QWEN3_DTYPE: {dtype_name}")

print(f"Loading model... (dtype={dtype_name})")
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    torch_dtype=model_dtype,
)
model.to(device).eval()
print(f"[Info] Model param dtype: model_dtype={model_dtype}, actual={model.dtype}")

try:
    from transformers.generation.configuration_utils import CompileConfig

    if getattr(model.generation_config, "compile_config", None) is None:
        model.generation_config.compile_config = CompileConfig()
except Exception:
    pass

if not hasattr(model.generation_config, "forced_decoder_ids"):
    model.generation_config.forced_decoder_ids = None

if not hasattr(model.config, "output_attentions"):
    model.config.output_attentions = False


def _cooldown(seconds: int) -> None:
    print(f"\n{'=' * 40}")
    print(f"[System] Cooling down for {seconds}s to stabilize CPU state...")
    for i in range(seconds, 0, -1):
        print(f"\rWaiting... {i}s remaining", end="", flush=True)
        time.sleep(1)
    print("\n[System] Ready! Starting benchmark.")

class TokenLatencyStreamer(BaseStreamer):
    def __init__(self, skip_prompt_len: int = 0) -> None:
        super().__init__()
        self.skip_prompt_len = skip_prompt_len  # 记录需要跳过的 Prompt 长度
        self.token_count = 0
        
        self.first_token_time: float | None = None
        self.last_token_time: float | None = None
        self.generated_tokens = 0

    def put(self, value) -> None:
        now = time.perf_counter()
        
        if isinstance(value, torch.Tensor):
            chunk_len = int(value.numel())
        else:
            chunk_len = len(value)
            
        # 核心修复逻辑：
        # 如果当前累计的 token 数还不足以覆盖 prompt，说明这批数据包含 prompt
        if self.token_count < self.skip_prompt_len:
            self.token_count += chunk_len
            # 如果这批数据刚好是 prompt（或者包含 prompt），我们不记录时间，直接返回
            # 注意：这里简化处理，假设 prompt 是一次性发来的（transformers 默认行为）
            return

        # 只有过了 Prompt 阶段，才开始计时和计数
        if self.generated_tokens == 0:
            self.first_token_time = now  # 这才是真正的首字生成时间 (Prefill 结束)
            
        self.last_token_time = now
        self.generated_tokens += chunk_len
        self.token_count += chunk_len

    def end(self) -> None:
        return


@pytest.mark.parametrize("enable_gems", [False, True])
def test_cpu_tps_and_first_token_latency(enable_gems):
    sleep_time = int(os.environ.get("FLAGGEMS_COOLDOWN", "10"))
    _cooldown(sleep_time)

    # Use fixed token length for stable prefill cost
    prompt_len = int(os.environ.get("QWEN3_PROMPT_TOKENS", "512"))
    input_ids = torch.randint(0, model.config.vocab_size, (1, prompt_len), device=device)
    input_len = input_ids.shape[1]
    inputs = {"input_ids": input_ids}
    print(f"[Info] input_len={input_len}")

    # Settings
    gen_tokens = 128
    repeats = 3
    beam_num = 1

    if enable_gems:
        run_context = flag_gems.use_gems()
        mode_name = "FlagGems"
    else:
        run_context = contextlib.nullcontext()
        mode_name = "Native PyTorch"

    print(f"[Info] Current Mode: {mode_name}")

    with run_context:
        # 1. Warmup
        print(f"[{mode_name}] Warming up...")
        with torch.no_grad():
            # model.generate(**inputs, max_new_tokens=2, min_new_tokens=2, num_beams=beam_num)
            model.generate(**inputs, max_new_tokens=1, num_beams=beam_num)
        
        print(f"[{mode_name}] Starting benchmark...")
        # 2. Benchmark Loop
        ttft_runs = []
        decode_tps_runs = []
        e2e_tps_runs = [] # Optional: End-to-End TPS
        
        total_generated_count = 0

        for i in range(repeats):
            streamer = TokenLatencyStreamer(skip_prompt_len=input_len)
            
            # Record absolute start time (before prefill)
            gen_start_time = time.perf_counter()
            
            with torch.no_grad():
                model.generate(
                    **inputs,
                    max_new_tokens=gen_tokens,
                    min_new_tokens=gen_tokens, # Force generation to ensure stable TPS
                    num_beams=beam_num,
                    streamer=streamer,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            # --- Calculation Logic ---
            
            # A. TTFT (Time To First Token)
            # Time from function call -> First token yielded
            if streamer.first_token_time is not None:
                ttft = streamer.first_token_time - gen_start_time
            else:
                ttft = 0.0 # Should not happen if gen_tokens > 0
            
            # B. Decode TPS (Generation Speed)
            # Formula: (N - 1) / (Time_Last - Time_First)
            # This excludes prefill time entirely.
            gen_duration = streamer.last_token_time - streamer.first_token_time
            if streamer.generated_tokens > 1 and gen_duration > 0:
                decode_tps = (streamer.generated_tokens - 1) / gen_duration
            else:
                decode_tps = 0.0

            # C. E2E TPS (Total Throughput including Prefill)
            total_duration = streamer.last_token_time - gen_start_time
            e2e_tps = streamer.generated_tokens / total_duration if total_duration > 0 else 0.0

            ttft_runs.append(ttft)
            decode_tps_runs.append(decode_tps)
            e2e_tps_runs.append(e2e_tps)
            
            total_generated_count = streamer.generated_tokens
            
            print(f"Run {i+1}: TTFT={ttft*1000:.1f}ms, Decode TPS={decode_tps:.2f}, E2E TPS={e2e_tps:.2f}")

        # Averages
        avg_ttft = sum(ttft_runs) / repeats
        avg_decode_tps = sum(decode_tps_runs) / repeats
        avg_e2e_tps = sum(e2e_tps_runs) / repeats

        print(f"\n{'-' * 40}")
        print(f"Mode               : {mode_name}")
        print(f"Avg TTFT           : {avg_ttft:.4f} s")
        print(f"Avg Decode TPS     : {avg_decode_tps:.2f} tokens/s (Key Metric)")
        print(f"Avg E2E TPS        : {avg_e2e_tps:.2f} tokens/s")
        print(f"Last Run Tokens    : {total_generated_count}")
        print(f"{'-' * 40}\n")
