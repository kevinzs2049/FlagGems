from __future__ import annotations

import functools
import os
from collections import Counter

ENV_AUTO_PATCH = "FLAGGEMS_VLLM_WORKER_AUTO_PATCH"
ENV_USE_ATEN = "FLAGGEMS_VLLM_WORKER_USE_ATEN"
ENV_USE_PATCH = "FLAGGEMS_VLLM_WORKER_USE_PATCH"
ENV_REGISTER_MODE = "FLAGGEMS_VLLM_WORKER_REGISTER_MODE"
ENV_EXCLUDE_OPS = "FLAGGEMS_VLLM_WORKER_EXCLUDE_OPS"
ENV_PATCH_VERBOSE = "FLAGGEMS_VLLM_WORKER_PATCH_VERBOSE"

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


def _make_register_class(use_all_ops):
    from flag_gems.runtime.register import Register

    if not use_all_ops:
        return Register
    return type(
        "NoExcludeRegister",
        (Register,),
        {"get_vendor_unused_op": lambda self: []},
    )


class FlagGemsVllmWorkerExtension:
    """Worker-side helper for applying and auditing FlagGems patches."""

    _patch_call_counter = Counter()
    _wrapped_patch_funcs = {}
    _auto_apply_attempted = False
    _auto_apply_ok = False
    _auto_apply_error = ""

    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls)
        cls._auto_apply_from_env()
        return instance

    @classmethod
    def _prepare_patch_trace(cls):
        import flag_gems.patches.patch_vllm_all as patch_mod

        for fn_name in TRACEABLE_PATCH_FUNCS:
            if fn_name in cls._wrapped_patch_funcs:
                continue
            original = getattr(patch_mod, fn_name, None)
            if original is None:
                continue

            @functools.wraps(original)
            def _wrapped(*args, __fn=original, __name=fn_name, **kwargs):
                cls._patch_call_counter[__name] += 1
                return __fn(*args, **kwargs)

            cls._wrapped_patch_funcs[fn_name] = _wrapped
            setattr(patch_mod, fn_name, _wrapped)

    @classmethod
    def _collect_patch_status(cls):
        import torch

        import flag_gems
        import flag_gems.patches.patch_vllm_all as patch_mod

        dispatch_key = flag_gems.runtime.device.dispatch_key
        is_cpu = dispatch_key == "CPU"
        status = {
            "dispatch_key": dispatch_key,
            "auto_apply_attempted": cls._auto_apply_attempted,
            "auto_apply_ok": cls._auto_apply_ok,
            "auto_apply_error": cls._auto_apply_error,
            "module_patch_installation": [],
            "lib_patch_dispatch_bound": [],
            "patch_fn_call_counts": {},
        }

        cpu_method_overrides = {
            ("RMSNorm", "forward_cuda"): "forward",
            ("RotaryEmbedding", "forward_cuda"): "forward",
            ("SiluAndMul", "forward_cuda"): "forward",
        }
        cpu_skip_targets = {
            ("TritonMLAImpl", "_forward_decode"),
            ("FlashAttentionImpl", "forward"),
            ("FlashAttnMLAImpl", "_forward_decode"),
        }

        for module_candidates, cls_name, method_name, fn_name in MODULE_PATCH_SPECS:
            effective_method = cpu_method_overrides.get(
                (cls_name, method_name), method_name
            ) if is_cpu else method_name
            expected_active = (cls_name, method_name) not in cpu_skip_targets if is_cpu else True
            patch_cls, module_name = _optional_import_first(module_candidates, cls_name)
            if patch_cls is None:
                status["module_patch_installation"].append(
                    {
                        "target": f"{module_candidates[0]}.{cls_name}.{effective_method}",
                        "status": "missing",
                        "patched_to_expected": False,
                        "expected_active": expected_active,
                        "resolved_module": None,
                        "expected_wrapper": fn_name,
                    }
                )
                continue
            current = getattr(patch_cls, effective_method, None)
            expected = cls._wrapped_patch_funcs.get(fn_name, getattr(patch_mod, fn_name, None))
            patched = expected is not None and current is expected if expected_active else None
            item_status = "ok" if expected_active else "skipped"
            status["module_patch_installation"].append(
                {
                    "target": f"{module_name}.{cls_name}.{effective_method}",
                    "status": item_status,
                    "patched_to_expected": bool(patched) if patched is not None else None,
                    "expected_active": expected_active,
                    "resolved_module": module_name,
                    "expected_wrapper": fn_name,
                    "current_qualname": getattr(current, "__qualname__", str(current)),
                }
            )

        cpu_active_libs = {
            "_C::silu_and_mul",
            "_C::apply_repetition_penalties_",
        }
        has_kernel_api = hasattr(torch._C, "_dispatch_has_kernel_for_dispatch_key")
        for lib_name, op_name, fn_name in LIB_PATCH_SPECS:
            qualified = f"{lib_name}::{op_name}"
            expected_active = qualified in cpu_active_libs if is_cpu else True
            bound = None
            err = None
            if expected_active:
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
            status["lib_patch_dispatch_bound"].append(
                {
                    "target": qualified,
                    "dispatch_key": dispatch_key,
                    "bound": bound,
                    "expected_active": expected_active,
                    "expected_wrapper": fn_name,
                    "error": err,
                }
            )

        for fn_name in TRACEABLE_PATCH_FUNCS:
            status["patch_fn_call_counts"][fn_name] = int(
                cls._patch_call_counter.get(fn_name, 0)
            )
        return status

    @classmethod
    def _parse_bool_env(cls, name, default=False):
        raw = os.getenv(name)
        if raw is None:
            return default
        return raw.strip().lower() in ("1", "true", "yes", "on")

    @classmethod
    def _auto_apply_from_env(cls):
        if cls._auto_apply_attempted:
            return
        cls._auto_apply_attempted = True

        if not cls._parse_bool_env(ENV_AUTO_PATCH, False):
            return

        try:
            use_aten = cls._parse_bool_env(ENV_USE_ATEN, True)
            use_patch = cls._parse_bool_env(ENV_USE_PATCH, True)
            register_mode = os.getenv(ENV_REGISTER_MODE, "default")
            exclude_raw = os.getenv(ENV_EXCLUDE_OPS, "")
            exclude_ops = [x.strip() for x in exclude_raw.split(",") if x.strip()]
            verbose = cls._parse_bool_env(ENV_PATCH_VERBOSE, False)
            # Apply once before worker warmup/compile.
            cls._apply_and_audit_impl(
                use_aten=use_aten,
                use_patch=use_patch,
                register_mode=register_mode,
                exclude_ops=exclude_ops,
                verbose=verbose,
            )
            cls._auto_apply_ok = True
            cls._auto_apply_error = ""
        except Exception as exc:
            cls._auto_apply_ok = False
            cls._auto_apply_error = str(exc)

    def flaggems_apply_and_audit(
        self,
        use_aten=True,
        use_patch=True,
        register_mode="default",
        exclude_ops=None,
        verbose=False,
    ):
        return self.__class__._apply_and_audit_impl(
            use_aten=use_aten,
            use_patch=use_patch,
            register_mode=register_mode,
            exclude_ops=exclude_ops,
            verbose=verbose,
        )

    @classmethod
    def _apply_and_audit_impl(
        cls,
        use_aten=True,
        use_patch=True,
        register_mode="default",
        exclude_ops=None,
        verbose=False,
    ):
        import flag_gems

        use_all_ops = register_mode == "all"
        exclude_ops = exclude_ops or []

        if use_aten:
            registrar = _make_register_class(use_all_ops)
            unused = exclude_ops if use_all_ops else (exclude_ops or None)
            flag_gems.enable(unused=unused, registrar=registrar)

        if use_patch:
            cls._prepare_patch_trace()
            flag_gems.apply_gems_patches_to_vllm(verbose=verbose)

        return cls._collect_patch_status()

    def flaggems_collect_patch_audit(self):
        return self.__class__._collect_patch_status()
