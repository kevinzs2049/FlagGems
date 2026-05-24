# ARM Qwen RMSNorm Triton Optimization Notes

## Background

In Qwen decode profiling on CPU/ARM (`GEMS_VENDOR=arm`), the RMSNorm math chain
was one of the dominant gaps between reference and gems:

- `pow`
- `mean`
- `rsqrt`
- `mul`/`add`

These ops were executed as separate kernels in the model path, creating extra
kernel launch and memory traffic overhead.

## Goal

Keep a Triton-first path and reduce runtime cost of RMSNorm-heavy decode steps by:

1. Replacing the decomposed Qwen RMSNorm math chain with one `flag_gems.rms_norm` call.
2. Keeping the feature optional and easy to A/B test.
3. Preserving original behavior outside supported conditions.

## Implementation Summary

### 1) Qwen RMSNorm runtime patch helper

Added: `examples/qwen_rmsnorm_patch.py`

- Patches both classes when present:
  - `transformers.models.qwen3.modeling_qwen3.Qwen3RMSNorm`
  - `transformers.models.qwen2.modeling_qwen2.Qwen2RMSNorm`
- New patched forward uses `flag_gems.rms_norm(...)` when:
  - tensor is on CPU
  - `weight.device == hidden_states.device`
  - `weight.dtype == hidden_states.dtype`
- Otherwise it falls back to the original Qwen RMSNorm math path.
- Guarded to avoid double patching:
  - `__flag_gems_triton_rmsnorm_patched__`

### 2) Qwen3 scripts integration (A/B switch)

Updated:

- `examples/model_qwen3_0_6b_test.py`
- `examples/model_qwen3_0_6b_walltime_bench.py`

Added env switch:

- `QWEN3_RMSNORM_TRITON_PATCH=1` to enable
- default is `0` (disabled)

When enabled, scripts print patched class count for traceability.

### 3) Add path safety fix

Updated: `src/flag_gems/runtime/backend/_arm/ops/add.py`

Refined one branch to avoid forcing the tensor-tensor Triton path for all
small same-shape tensors. The path now requires both tensors to be contiguous:

- before: same shape + small size (`numel <= 8192`) could enter this path
- after: same shape and both contiguous only

This avoids unnecessary layout normalization overhead in non-contiguous cases.

## Reproduce

Environment:

```bash
source ~/venvdebug/bin/activate
export PYTHONPATH=$PWD/src
export GEMS_VENDOR=arm
```

Profiling A/B:

```bash
QWEN3_RMSNORM_TRITON_PATCH=0 pytest examples/model_qwen3_0_6b_test.py -k test_accuracy_Qwen -s
QWEN3_RMSNORM_TRITON_PATCH=1 pytest examples/model_qwen3_0_6b_test.py -k test_accuracy_Qwen -s
```

Walltime A/B:

```bash
QWEN3_RMSNORM_TRITON_PATCH=0 QWEN3_BENCH_WARMUP=0 QWEN3_BENCH_REPEATS=2 QWEN3_MAX_NEW_TOKENS=32 python examples/model_qwen3_0_6b_walltime_bench.py
QWEN3_RMSNORM_TRITON_PATCH=1 QWEN3_BENCH_WARMUP=0 QWEN3_BENCH_REPEATS=2 QWEN3_MAX_NEW_TOKENS=32 python examples/model_qwen3_0_6b_walltime_bench.py
```

## Observed Results (Current Machine)

### Profiling run (`model_qwen3_0_6b_test.py`)

- OFF (`QWEN3_RMSNORM_TRITON_PATCH=0`):
  - gems elapsed: `3.560s`
  - key gaps:
    - `aten::add` `265.44ms`
    - `aten::mean` `194.32ms`
    - `aten::pow` `112.12ms`
    - `aten::rsqrt` `69.27ms`
    - `aten::neg` `75.33ms`

- ON (`QWEN3_RMSNORM_TRITON_PATCH=1`):
  - gems elapsed: `3.132s`
  - RMSNorm appears as a single `RmsNorm` op (instead of split chain)
  - `aten::add` reduced to `186.79ms`
  - `pow/mean/rsqrt` no longer dominate as separate hotspots

### Walltime run (`model_qwen3_0_6b_walltime_bench.py`, repeats=2)

- OFF:
  - gems `tok/s`: `3.134`
- ON:
  - gems `tok/s`: `3.182`

Interpretation:

- Profiling path shows clear improvement.
- No-profiler walltime also improves, but by a smaller margin in this short run.

## Risk and Scope

- Patch is runtime-only and gated by env var.
- Default behavior is unchanged (`QWEN3_RMSNORM_TRITON_PATCH=0`).
- Non-CPU or dtype/device mismatch cases fall back to original Qwen implementation.

## Next Steps

1. Re-run ON/OFF with more repeats and longer prompt/token settings to reduce noise.
2. Apply the same patch switch for Qwen2 benchmark scripts and compare with the same protocol.
3. Continue shape-bucket optimization on remaining high-gap ops (`neg`, `masked_fill_`, `index` chain).
