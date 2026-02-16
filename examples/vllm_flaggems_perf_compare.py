import argparse
import functools
import json
import os
import statistics
import subprocess
import sys
import time
from collections import Counter


DEFAULT_PROMPTS = [
    "What is your name?",
    "Explain gradient descent in simple terms.",
    "Write a short poem about autumn moonlight.",
]


def _parse_ops_csv(raw):
    return [item.strip() for item in raw.split(",") if item.strip()]


def _env_flag(name, default=False):
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _ensure_vllm_tracer_symbol():
    # Some vLLM builds expect `vllm.tracing.Tracer` but only expose it in
    # `vllm.tracing.otel`. Add the symbol before importing `vllm.LLM`.
    try:
        import vllm.tracing as tracing_mod

        if not hasattr(tracing_mod, "Tracer"):
            from vllm.tracing.otel import Tracer as otel_tracer

            tracing_mod.Tracer = otel_tracer
    except Exception:
        # Keep default behavior if tracing internals are unavailable.
        pass


def _ensure_vllm_cpu_platform():
    # Some local editable vLLM layouts fail auto platform detection and leave
    # `device_type` empty. Pin to CPU so benchmark runs in the expected backend.
    try:
        import vllm.platforms as platforms_mod

        current = platforms_mod.current_platform
        if getattr(current, "device_type", ""):
            return
        from vllm.platforms.cpu import CpuPlatform

        platforms_mod.current_platform = CpuPlatform()
    except Exception:
        pass


def _build_parser():
    parser = argparse.ArgumentParser(
        description="Compare vLLM performance with/without FlagGems patches."
    )
    parser.add_argument(
        "--model",
        default=os.getenv("VLLM_MODEL_PATH", "Qwen/Qwen2.5-7B-Instruct"),
        help="Model name or local model path.",
    )
    parser.add_argument(
        "--prompts",
        default=os.getenv("VLLM_PROMPTS", "||".join(DEFAULT_PROMPTS)),
        help="Prompts joined by '||'.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=int(os.getenv("VLLM_MAX_NEW_TOKENS", "32")),
        help="Generation length per prompt.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=float(os.getenv("VLLM_TEMPERATURE", "0.0")),
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=float(os.getenv("VLLM_TOP_P", "1.0")),
        help="Top-p sampling value.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=int(os.getenv("VLLM_REPEATS", "3")),
        help="Measured runs after warmup for each mode.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=int(os.getenv("VLLM_WARMUP", "1")),
        help="Warmup runs per mode.",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=int(os.getenv("VLLM_TP", "1")),
        help="vLLM tensor parallel size.",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=int(os.getenv("VLLM_MAX_MODEL_LEN", "1024")),
        help="vLLM max model length.",
    )
    parser.add_argument(
        "--dtype",
        default=os.getenv("VLLM_DTYPE", "auto"),
        help="vLLM dtype, e.g. auto/float16/bfloat16.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True to vLLM.",
    )
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        help="Pass enforce_eager=True to vLLM.",
    )
    parser.add_argument(
        "--patch-verbose",
        action="store_true",
        help="Print detailed patch logs from apply_gems_patches_to_vllm.",
    )
    parser.add_argument(
        "--modes",
        default=os.getenv("VLLM_BENCH_MODES", "ref,gems_aten,gems_full"),
        help=(
            "Comma-separated modes: ref, gems_aten, gems_patch_only, gems_full. "
            "Default compares ref/gems_aten/gems_full."
        ),
    )
    parser.add_argument(
        "--gems-exclude-ops",
        default=os.getenv("VLLM_GEMS_EXCLUDE_OPS", ""),
        help="Comma-separated FlagGems ATen op names to exclude in gems_aten/gems_full.",
    )
    parser.add_argument(
        "--gems-register-mode",
        choices=("default", "all"),
        default=os.getenv("VLLM_GEMS_REGISTER_MODE", "default"),
        help=(
            "FlagGems registration mode. "
            "'default' keeps ARM vendor exclude list, "
            "'all' disables vendor exclude list."
        ),
    )
    parser.add_argument(
        "--audit-coverage",
        action="store_true",
        default=_env_flag("VLLM_AUDIT_COVERAGE", False),
        help=(
            "Collect operator coverage evidence (aten bases covered by FlagGems "
            "and per-key call counts). Adds one extra profiled generate per mode."
        ),
    )
    parser.add_argument(
        "--child-mode",
        default=None,
        help=argparse.SUPPRESS,
    )
    return parser


def _normalize_prompts(raw):
    prompts = [x.strip() for x in raw.split("||") if x.strip()]
    return prompts or DEFAULT_PROMPTS


def _collect_generated_tokens(outputs):
    total = 0
    for item in outputs:
        if item.outputs:
            total += len(item.outputs[0].token_ids)
    return total


class _CountingRegisterMixin:
    called_keys = Counter()
    called_funcs = Counter()

    @classmethod
    def clear_counters(cls):
        cls.called_keys.clear()
        cls.called_funcs.clear()

    def register_impl(self, key, fn):
        @functools.wraps(fn)
        def wrapped(*args, **kwargs):
            self.__class__.called_keys[key] += 1
            self.__class__.called_funcs[fn.__name__] += 1
            return fn(*args, **kwargs)

        return super().register_impl(key, wrapped)


def _make_register_class(use_all_ops, with_counter):
    from flag_gems.runtime.register import Register

    base = Register
    if with_counter:
        base = type("CountingRegisterBase", (_CountingRegisterMixin, Register), {})
    if use_all_ops:
        return type(
            "NoExcludeRegister",
            (base,),
            {"get_vendor_unused_op": lambda self: []},
        )
    return base


def _collect_coverage(prof, register_cls):
    aten_bases = set()
    for evt in prof.key_averages():
        name = evt.key
        if name.startswith("aten::"):
            aten_bases.add(name[len("aten::") :].split(".")[0])

    if not hasattr(register_cls, "called_keys"):
        return {
            "aten_base_count": len(aten_bases),
            "flaggems_registered_key_count": 0,
            "flaggems_called_key_count": 0,
            "flaggems_total_calls": 0,
            "covered_aten_base_count": 0,
            "uncovered_aten_base_count": len(aten_bases),
            "covered_aten_bases": [],
            "uncovered_aten_bases": sorted(aten_bases),
            "flaggems_called_topk": [],
        }

    fg_keys = register_cls.called_keys
    fg_bases = {k.split(".")[0] for k in fg_keys}
    covered = sorted(aten_bases & fg_bases)
    uncovered = sorted(aten_bases - fg_bases)
    topk = [
        {"key": key, "calls": count}
        for key, count in fg_keys.most_common(80)
    ]

    return {
        "aten_base_count": len(aten_bases),
        "flaggems_called_key_count": len(fg_keys),
        "flaggems_total_calls": int(sum(fg_keys.values())),
        "covered_aten_base_count": len(covered),
        "uncovered_aten_base_count": len(uncovered),
        "covered_aten_bases": covered,
        "uncovered_aten_bases": uncovered,
        "flaggems_called_topk": topk,
    }


def _run_single_mode(args, mode):
    prompts = _normalize_prompts(args.prompts)
    gems_exclude_ops = _parse_ops_csv(args.gems_exclude_ops)

    if mode not in {"ref", "gems_aten", "gems_patch_only", "gems_full"}:
        raise ValueError(f"Unsupported mode: {mode}")

    register_cls = None
    register_mode = args.gems_register_mode
    register_all = register_mode == "all"
    coverage = None

    if mode != "ref":
        import flag_gems

        if mode in {"gems_aten", "gems_full"}:
            register_cls = _make_register_class(
                use_all_ops=register_all,
                with_counter=args.audit_coverage,
            )
            exclude_setting = gems_exclude_ops if register_all else (gems_exclude_ops or None)
            if hasattr(register_cls, "clear_counters"):
                register_cls.clear_counters()
            flag_gems.enable(unused=exclude_setting, registrar=register_cls)
        if mode in {"gems_patch_only", "gems_full"}:
            flag_gems.apply_gems_patches_to_vllm(verbose=args.patch_verbose)

    _ensure_vllm_tracer_symbol()
    _ensure_vllm_cpu_platform()
    from vllm import LLM, SamplingParams

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
    )

    llm_kwargs = {
        "model": args.model,
        "tensor_parallel_size": args.tensor_parallel_size,
        "max_model_len": args.max_model_len,
        "enforce_eager": args.enforce_eager,
        "trust_remote_code": args.trust_remote_code,
    }
    if args.dtype and args.dtype != "auto":
        llm_kwargs["dtype"] = args.dtype

    llm = LLM(**llm_kwargs)

    for _ in range(args.warmup):
        llm.generate(prompts, sampling_params)

    if register_cls is not None and hasattr(register_cls, "clear_counters"):
        register_cls.clear_counters()

    elapsed_list = []
    tok_per_s_list = []
    generated_tokens_list = []
    for _ in range(args.repeats):
        start = time.perf_counter()
        outputs = llm.generate(prompts, sampling_params)
        elapsed_s = time.perf_counter() - start
        generated_tokens = _collect_generated_tokens(outputs)
        tok_per_s = generated_tokens / elapsed_s if elapsed_s > 0 else 0.0
        elapsed_list.append(elapsed_s)
        tok_per_s_list.append(tok_per_s)
        generated_tokens_list.append(generated_tokens)

    if args.audit_coverage and mode != "ref":
        import torch

        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU],
            record_shapes=False,
            profile_memory=False,
        ) as prof:
            llm.generate(prompts, sampling_params)
        coverage = _collect_coverage(prof, register_cls)
        if register_cls is not None:
            coverage["flaggems_registered_key_count"] = len(
                flag_gems.all_registered_keys()
            )
        else:
            coverage["flaggems_registered_key_count"] = 0

    result = {
        "mode": mode,
        "model": args.model,
        "prompt_count": len(prompts),
        "max_new_tokens": args.max_new_tokens,
        "repeats": args.repeats,
        "warmup": args.warmup,
        "elapsed_s_avg": statistics.mean(elapsed_list),
        "elapsed_s_min": min(elapsed_list),
        "tok_per_s_avg": statistics.mean(tok_per_s_list),
        "tok_per_s_max": max(tok_per_s_list),
        "generated_tokens_avg": statistics.mean(generated_tokens_list),
        "gems_exclude_ops": gems_exclude_ops,
        "gems_register_mode": register_mode,
    }
    if coverage is not None:
        result["coverage"] = coverage
    return result


def _to_child_argv(args, mode):
    argv = [
        sys.executable,
        __file__,
        "--child-mode",
        mode,
        "--model",
        args.model,
        "--prompts",
        args.prompts,
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--temperature",
        str(args.temperature),
        "--top-p",
        str(args.top_p),
        "--repeats",
        str(args.repeats),
        "--warmup",
        str(args.warmup),
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
        "--max-model-len",
        str(args.max_model_len),
        "--dtype",
        str(args.dtype),
        "--gems-exclude-ops",
        args.gems_exclude_ops,
        "--gems-register-mode",
        args.gems_register_mode,
    ]
    if args.trust_remote_code:
        argv.append("--trust-remote-code")
    if args.enforce_eager:
        argv.append("--enforce-eager")
    if args.patch_verbose:
        argv.append("--patch-verbose")
    if args.audit_coverage:
        argv.append("--audit-coverage")
    return argv


def _run_parent(args):
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    if not modes:
        raise ValueError("At least one mode is required.")

    results = []
    for mode in modes:
        child_cmd = _to_child_argv(args, mode)
        print(f"[vllm-bench] running mode={mode}")
        proc = subprocess.run(
            child_cmd,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        print(proc.stdout, end="")
        if proc.returncode != 0:
            raise RuntimeError(
                f"Child benchmark failed for mode={mode} with exit code={proc.returncode}"
            )
        marker = "[vllm-bench-json]"
        payload = None
        for line in proc.stdout.splitlines():
            if line.startswith(marker):
                payload = line[len(marker) :]
        if payload is None:
            raise RuntimeError(f"No benchmark json found for mode={mode}")
        results.append(json.loads(payload))

    print("\n[vllm-bench] summary")
    print(
        "mode               elapsed_avg(s)  elapsed_min(s)  tok/s_avg  "
        "tok/s_max  gen_tokens_avg"
    )
    for row in results:
        print(
            f"{row['mode']:<18} {row['elapsed_s_avg']:<14.3f} "
            f"{row['elapsed_s_min']:<14.3f} {row['tok_per_s_avg']:<9.3f} "
            f"{row['tok_per_s_max']:<9.3f} {row['generated_tokens_avg']:.1f}"
        )

    ref = next((x for x in results if x["mode"] == "ref"), None)
    if ref:
        print("\n[vllm-bench] speedup vs ref (tok/s_avg)")
        for row in results:
            if row["mode"] == "ref":
                continue
            speedup = row["tok_per_s_avg"] / ref["tok_per_s_avg"]
            print(f"{row['mode']}: {speedup:.3f}x")

    covered_rows = [r for r in results if "coverage" in r]
    if covered_rows:
        print("\n[vllm-bench] coverage")
        for row in covered_rows:
            cov = row["coverage"]
            print(
                f"{row['mode']}: register_mode={row.get('gems_register_mode', 'default')} "
                f"registered={cov['flaggems_registered_key_count']} "
                f"called_keys={cov['flaggems_called_key_count']} "
                f"total_calls={cov['flaggems_total_calls']} "
                f"covered_aten={cov['covered_aten_base_count']}/{cov['aten_base_count']} "
                f"uncovered={cov['uncovered_aten_base_count']}"
            )
            print(
                f"{row['mode']}: uncovered_aten={','.join(cov['uncovered_aten_bases'][:60])}"
            )


def main():
    parser = _build_parser()
    args = parser.parse_args()

    if args.child_mode:
        result = _run_single_mode(args, args.child_mode)
        print("[vllm-bench-json]" + json.dumps(result, sort_keys=True))
        return

    _run_parent(args)


if __name__ == "__main__":
    main()
