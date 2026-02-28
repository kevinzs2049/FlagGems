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

MODULE_PATCH_SPECS = [
    (
        ["vllm.model_executor.layers.layernorm"],
        "RMSNorm",
        "forward_cuda",
        "custom_gems_rms_forward_cuda",
    ),
    (
        ["vllm.model_executor.layers.rotary_embedding"],
        "RotaryEmbedding",
        "forward_cuda",
        "custom_gems_rope_forward_cuda",
    ),
    (
        ["vllm.attention.ops.paged_attn", "vllm.v1.attention.ops.paged_attn"],
        "PagedAttention",
        "write_to_paged_cache",
        "custom_gems_write_to_paged_cache",
    ),
    (
        ["vllm.model_executor.layers.activation"],
        "SiluAndMul",
        "forward_cuda",
        "custom_gems_silu_and_mul",
    ),
    (
        ["vllm.v1.attention.backends.mla.triton_mla"],
        "TritonMLAImpl",
        "_forward_decode",
        "custom_gems_flash_mla_forward",
    ),
    (
        ["vllm.v1.attention.backends.flash_attn"],
        "FlashAttentionImpl",
        "forward",
        "custom_gems_flash_attention_impl_forward",
    ),
    (
        ["vllm.v1.attention.backends.mla.flashattn_mla"],
        "FlashAttnMLAImpl",
        "_forward_decode",
        "custom_gems_flashattn_mla_forward_decode",
    ),
]

LIB_PATCH_SPECS = [
    ("_C", "silu_and_mul", "custom_silu_and_mul"),
    ("_C", "cutlass_scaled_mm", "custom_cutlass_scaled_mm"),
    ("_moe_C", "moe_align_block_size", "custom_moe_align_block_size"),
    ("_moe_C", "topk_softmax", "custom_topk_softmax"),
    ("_moe_C", "moe_sum", "custom_moe_sum"),
    ("_vllm_fa3_C", "get_scheduler_metadata", "custom_get_scheduler_metadata"),
    ("_moe_C", "grouped_topk", "custom_moe_grouped_topk"),
    ("_C", "per_token_group_fp8_quant", "custom_per_token_group_fp8_quant"),
    ("_C", "apply_repetition_penalties_", "custom_apply_repetition_penalties"),
    ("_C_cache_ops", "concat_and_cache_mla", "custom_concat_and_cache_mla"),
]

TRACEABLE_PATCH_FUNCS = sorted(
    {spec[3] for spec in MODULE_PATCH_SPECS} | {spec[2] for spec in LIB_PATCH_SPECS}
)

FLAGGEMS_VLLM_WORKER_EXTENSION = (
    "flag_gems.patches.vllm_worker_extension.FlagGemsVllmWorkerExtension"
)
ENV_WORKER_AUTO_PATCH = "FLAGGEMS_VLLM_WORKER_AUTO_PATCH"
ENV_WORKER_USE_ATEN = "FLAGGEMS_VLLM_WORKER_USE_ATEN"
ENV_WORKER_USE_PATCH = "FLAGGEMS_VLLM_WORKER_USE_PATCH"
ENV_WORKER_REGISTER_MODE = "FLAGGEMS_VLLM_WORKER_REGISTER_MODE"
ENV_WORKER_EXCLUDE_OPS = "FLAGGEMS_VLLM_WORKER_EXCLUDE_OPS"
ENV_WORKER_PATCH_VERBOSE = "FLAGGEMS_VLLM_WORKER_PATCH_VERBOSE"


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
        "--output-json",
        default=os.getenv("VLLM_BENCH_OUTPUT_JSON", ""),
        help="Optional path to save aggregate benchmark/audit JSON from parent process.",
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


def _collect_profiler_summary(prof, top_k=120):
    rows = []
    category_self_ms = Counter()
    for evt in prof.key_averages():
        key = evt.key
        self_ms = float(evt.self_cpu_time_total) / 1000.0
        total_ms = float(evt.cpu_time_total) / 1000.0
        count = int(evt.count)
        if key.startswith("aten::"):
            category = "aten"
        elif key.startswith("flag_gems::"):
            category = "flag_gems_custom"
        elif key.startswith("vllm::"):
            category = "vllm_custom"
        else:
            category = "other"
        category_self_ms[category] += self_ms
        rows.append(
            {
                "key": key,
                "count": count,
                "self_cpu_ms": round(self_ms, 4),
                "cpu_total_ms": round(total_ms, 4),
                "category": category,
            }
        )
    rows.sort(key=lambda x: x["self_cpu_ms"], reverse=True)
    return {
        "top_ops": rows[:top_k],
        "category_self_cpu_ms": {
            k: round(v, 4) for k, v in sorted(category_self_ms.items())
        },
    }


def _optional_import_attr(module_name, attr_name):
    try:
        module = __import__(module_name, fromlist=[attr_name])
        return getattr(module, attr_name)
    except Exception:
        return None


def _optional_import_first(candidates, attr_name):
    for module_name in candidates:
        obj = _optional_import_attr(module_name, attr_name)
        if obj is not None:
            return obj, module_name
    return None, None


def _prepare_vllm_patch_trace():
    import flag_gems.patches.patch_vllm_all as patch_mod

    counter = Counter()
    wrapped = {}
    for fn_name in TRACEABLE_PATCH_FUNCS:
        original = getattr(patch_mod, fn_name, None)
        if original is None:
            continue

        @functools.wraps(original)
        def _wrapped(*args, __fn=original, __name=fn_name, **kwargs):
            counter[__name] += 1
            return __fn(*args, **kwargs)

        setattr(patch_mod, fn_name, _wrapped)
        wrapped[fn_name] = _wrapped
    return {"counter": counter, "wrapped": wrapped}


def _collect_vllm_patch_evidence(trace_state, dispatch_key):
    evidence = {
        "module_patch_installation": [],
        "lib_patch_dispatch_bound": [],
        "patch_fn_call_counts": {},
    }
    wrapped = trace_state.get("wrapped", {})

    for module_candidates, cls_name, method_name, fn_name in MODULE_PATCH_SPECS:
        cls, module_name = _optional_import_first(module_candidates, cls_name)
        if cls is None:
            evidence["module_patch_installation"].append(
                {
                    "target": f"{module_candidates[0]}.{cls_name}.{method_name}",
                    "status": "missing",
                    "patched_to_expected": False,
                    "resolved_module": None,
                    "expected_wrapper": fn_name,
                }
            )
            continue
        current = getattr(cls, method_name, None)
        expected = wrapped.get(fn_name)
        patched = expected is not None and current is expected
        evidence["module_patch_installation"].append(
            {
                "target": f"{module_name}.{cls_name}.{method_name}",
                "status": "ok",
                "patched_to_expected": bool(patched),
                "resolved_module": module_name,
                "expected_wrapper": fn_name,
                "current_qualname": getattr(current, "__qualname__", str(current)),
            }
        )

    import torch

    has_kernel_api = hasattr(torch._C, "_dispatch_has_kernel_for_dispatch_key")
    for lib_name, op_name, fn_name in LIB_PATCH_SPECS:
        qualified = f"{lib_name}::{op_name}"
        bound = None
        err = None
        try:
            if has_kernel_api:
                bound = bool(
                    torch._C._dispatch_has_kernel_for_dispatch_key(
                        qualified, dispatch_key
                    )
                )
            else:
                table = torch._C._dispatch_dump_table(qualified)
                bound = dispatch_key in table
        except Exception as exc:
            err = str(exc)
        evidence["lib_patch_dispatch_bound"].append(
            {
                "target": qualified,
                "dispatch_key": dispatch_key,
                "bound": bound,
                "expected_wrapper": fn_name,
                "error": err,
            }
        )

    for fn_name in TRACEABLE_PATCH_FUNCS:
        evidence["patch_fn_call_counts"][fn_name] = int(trace_state["counter"].get(fn_name, 0))
    return evidence


def _run_single_mode(args, mode):
    prompts = _normalize_prompts(args.prompts)
    gems_exclude_ops = _parse_ops_csv(args.gems_exclude_ops)

    if mode not in {"ref", "gems_aten", "gems_patch_only", "gems_full"}:
        raise ValueError(f"Unsupported mode: {mode}")

    register_cls = None
    register_mode = args.gems_register_mode
    register_all = register_mode == "all"
    coverage = None
    patch_trace_state = None
    worker_patch_apply_status = None
    worker_patch_final_status = None
    use_aten = False
    use_patch = False
    worker_auto_patch_before_warmup = False

    if mode != "ref":
        import flag_gems

        if mode in {"gems_aten", "gems_full"}:
            use_aten = True
            register_cls = _make_register_class(
                use_all_ops=register_all,
                with_counter=args.audit_coverage,
            )
            exclude_setting = gems_exclude_ops if register_all else (gems_exclude_ops or None)
            if hasattr(register_cls, "clear_counters"):
                register_cls.clear_counters()
            flag_gems.enable(unused=exclude_setting, registrar=register_cls)
        if mode in {"gems_patch_only", "gems_full"}:
            use_patch = True
            if args.audit_coverage:
                patch_trace_state = _prepare_vllm_patch_trace()
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
    if mode != "ref":
        # CPU backend executes model code in spawned worker processes.
        # Inject worker extension so we can apply/verify FlagGems from workers.
        llm_kwargs["worker_extension_cls"] = FLAGGEMS_VLLM_WORKER_EXTENSION
    if args.dtype and args.dtype != "auto":
        llm_kwargs["dtype"] = args.dtype

    env_backup = {}
    if mode != "ref":
        worker_auto_patch_before_warmup = _env_flag(
            "VLLM_WORKER_AUTO_PATCH_BEFORE_WARMUP", args.enforce_eager
        )
        worker_env = {
            ENV_WORKER_AUTO_PATCH: "1" if worker_auto_patch_before_warmup else "0",
            ENV_WORKER_USE_ATEN: "1" if use_aten else "0",
            ENV_WORKER_USE_PATCH: "1" if use_patch else "0",
            ENV_WORKER_REGISTER_MODE: register_mode,
            ENV_WORKER_EXCLUDE_OPS: ",".join(gems_exclude_ops),
            ENV_WORKER_PATCH_VERBOSE: "1" if args.patch_verbose else "0",
        }
        for key, value in worker_env.items():
            env_backup[key] = os.environ.get(key)
            os.environ[key] = value

    llm = LLM(**llm_kwargs)

    if mode != "ref":
        try:
            if worker_auto_patch_before_warmup:
                worker_patch_apply_status = llm.llm_engine.collective_rpc(
                    "flaggems_collect_patch_audit"
                )
            else:
                worker_patch_apply_status = llm.llm_engine.collective_rpc(
                    "flaggems_apply_and_audit",
                    kwargs={
                        "use_aten": use_aten,
                        "use_patch": use_patch,
                        "register_mode": register_mode,
                        "exclude_ops": gems_exclude_ops,
                        "verbose": args.patch_verbose,
                    },
                )
        except Exception as exc:
            worker_patch_apply_status = {"error": str(exc)}
        for key, old_val in env_backup.items():
            if old_val is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_val

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
        coverage["profiler"] = _collect_profiler_summary(prof)
        if not coverage["profiler"]["top_ops"]:
            coverage["profiler"]["note"] = (
                "No CPU events captured in this process. "
                "For vLLM CPU, model execution may run in spawned worker processes."
            )
        if register_cls is not None:
            coverage["flaggems_registered_key_count"] = len(
                flag_gems.all_registered_keys()
            )
        else:
            coverage["flaggems_registered_key_count"] = 0
        if patch_trace_state is not None:
            coverage["vllm_patch_evidence"] = _collect_vllm_patch_evidence(
                patch_trace_state,
                flag_gems.runtime.device.dispatch_key,
            )
        try:
            worker_patch_final_status = llm.llm_engine.collective_rpc(
                "flaggems_collect_patch_audit"
            )
        except Exception as exc:
            worker_patch_final_status = {"error": str(exc)}
        if (
            isinstance(worker_patch_final_status, list)
            and worker_patch_final_status
            and not args.enforce_eager
        ):
            calls = worker_patch_final_status[0].get("patch_fn_call_counts", {})
            if calls and not any(v > 0 for v in calls.values()):
                worker_patch_final_status[0]["note"] = (
                    "Patch installed but no patch calls observed. "
                    "With CPU spawn workers, warmup/compile may happen before patch RPC; "
                    "try --enforce-eager for runtime call validation."
                )
        coverage["worker_patch_apply_status"] = worker_patch_apply_status
        coverage["worker_patch_final_status"] = worker_patch_final_status

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
            patch_ev = cov.get("vllm_patch_evidence")
            if patch_ev:
                module_ok = sum(
                    1
                    for item in patch_ev["module_patch_installation"]
                    if item["patched_to_expected"]
                )
                module_total = len(patch_ev["module_patch_installation"])
                lib_ok = sum(
                    1
                    for item in patch_ev["lib_patch_dispatch_bound"]
                    if item["bound"] is True
                )
                lib_total = len(patch_ev["lib_patch_dispatch_bound"])
                print(
                    f"{row['mode']}: vllm_patch module_ok={module_ok}/{module_total} "
                    f"lib_dispatch_bound={lib_ok}/{lib_total}"
                )
                hot_patch = sorted(
                    patch_ev["patch_fn_call_counts"].items(),
                    key=lambda x: x[1],
                    reverse=True,
                )[:12]
                print(
                    f"{row['mode']}: vllm_patch_call_top={','.join(f'{k}:{v}' for k, v in hot_patch if v > 0)}"
                )
            worker_status = cov.get("worker_patch_final_status")
            if isinstance(worker_status, list) and worker_status:
                ws = worker_status[0]
                active_module_items = [
                    item
                    for item in ws.get("module_patch_installation", [])
                    if item.get("expected_active", True)
                ]
                module_ok = sum(
                    1
                    for item in active_module_items
                    if item.get("patched_to_expected")
                )
                module_total = len(active_module_items)
                active_lib_items = [
                    item
                    for item in ws.get("lib_patch_dispatch_bound", [])
                    if item.get("expected_active", True)
                ]
                lib_ok = sum(
                    1
                    for item in active_lib_items
                    if item.get("bound") is True
                )
                lib_total = len(active_lib_items)
                print(
                    f"{row['mode']}: worker_patch module_ok={module_ok}/{module_total} "
                    f"lib_dispatch_bound={lib_ok}/{lib_total}"
                )
                worker_hot = sorted(
                    ws.get("patch_fn_call_counts", {}).items(),
                    key=lambda x: x[1],
                    reverse=True,
                )[:12]
                print(
                    f"{row['mode']}: worker_patch_call_top={','.join(f'{k}:{v}' for k, v in worker_hot if v > 0)}"
                )
            elif worker_status:
                print(f"{row['mode']}: worker_patch_error={worker_status}")

    if args.output_json:
        payload = {
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "args": vars(args),
            "results": results,
        }
        with open(args.output_json, "w") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        print(f"\n[vllm-bench] wrote json: {args.output_json}")


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
