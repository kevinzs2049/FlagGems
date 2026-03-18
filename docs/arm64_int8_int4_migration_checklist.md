# ARM64 Int8/Int4 Migration Checklist (FlagGems + vLLM)

## 1) Baseline facts from current code

- FlagGems `mm` is float-oriented (`float16/bfloat16/float32`), not generic int8 GEMM-first:
  - `src/flag_gems/ops/mm.py`
- FlagGems int8 high-performance path is currently tied to `cutlass_scaled_mm` and NVIDIA SM dispatch:
  - `src/flag_gems/fused/cutlass_scaled_mm.py`
  - int8 assertion in `dispatch_scaled_mm(...)`
  - SM80/SM89/SM100/SM120 stubs are still `NotImplementedError`
- Quantized tensor coverage is partial and includes explicit fallback / unsupported paths:
  - `src/flag_gems/ops/copy.py` (`is_quantized` -> ATen redispatch)
  - `src/flag_gems/ops/to.py` (`is_quantized` -> `NotImplementedError`)
  - `src/flag_gems/ops/isclose.py` (`is_quantized` -> `RuntimeError`)
- In current FlagGems tree, no native int4 operator implementation is exposed under `src/flag_gems`.

## 2) ARM64 int4 opportunity signals (from vLLM CPU code)

- vLLM already has a CPU 4-bit MoE path wired as a custom op:
  - `/home/kevin/vllm_source/csrc/moe/dynamic_4bit_int_moe_cpu.cpp`
  - `/home/kevin/vllm_source/csrc/cpu/torch_bindings.cpp`
- The implementation explicitly targets AArch64 via `at::_ops::_dyn_quant_matmul_4bit`:
  - `#if defined(__aarch64__)` guard in `dynamic_4bit_int_moe_cpu.cpp`
- This is a strong migration anchor: ARM64 already has a usable 4-bit matmul primitive path in the stack, so FlagGems can align on operator contracts/packing formats before designing new kernels.

## 3) Priority roadmap

### P0 (must do first): define boundaries and evidence

- [ ] Freeze a target operator surface for quantized ARM64 first wave:
  - `mm`, `addmm`, `linear`, `index_select`-related gather path, `silu_and_mul` side chain.
- [ ] Add runtime evidence hooks for vLLM+FlagGems (CSV-ready):
  - per-op call count
  - per-op total self time
  - backend tag (`triton_arm`, `aten_fallback`, `vllm_custom`)
- [ ] Keep attention scope fixed:
  - exclude flash attention rewrite in this phase (to avoid confounding bottlenecks).

### P1 (int8 first): make ARM64 quantized linear path real

- [ ] Add `_arm`-scoped int8 `mm`/`addmm` implementations (do not touch generic path):
  - `src/flag_gems/runtime/backend/_arm/ops/mm.py`
  - `src/flag_gems/runtime/backend/_arm/ops/addmm.py`
- [ ] Define explicit dtype + layout contracts (row/col major, group size, accumulator dtype).
- [ ] Add dequant fusion strategy:
  - int8 weight + scale in one kernel launch where possible.
- [ ] Add strict fallback policy:
  - unsupported shape/dtype -> deterministic fallback + warning counter (not silent).

### P2 (int4 next): weight-only path with decode-first shapes

- [ ] Implement ARM64 `_arm` int4 pack/depack utility kernels (nibble packing, group scale layout).
- [ ] Build `int4_weight_only_mm` / `int4_addmm` in `_arm/ops` with decode-biased tuning:
  - prioritize `M=1` / small-`M`, large-`K/N`.
- [ ] Add model-facing gate in vLLM patch path:
  - only enable for validated shape/dtype/layout buckets.
- [ ] Reuse vLLM CPU dynamic 4-bit MoE shape conventions where possible to reduce integration friction.

### P3 (integration): vLLM + FlagGems quantized path closure

- [ ] Extend `patch_vllm_all.py` patch map for quantized linear hotspots used in your model configs.
- [ ] Add A/B harness:
  - `vllm native`
  - `vllm + FlagGems (quantized path on)`
  - same prompts/tokens/seeds and warmup policy.
- [ ] Export aggregate + per-op diff artifacts per run.

## 4) Int8/Int4 risk matrix

- **Kernel correctness risk (high)**: quantized scale/zero-point semantics mismatch.
  - Mitigation: op-level numerical golden tests per shape bucket.
- **Layout mismatch risk (high)**: packed weight format mismatch between loader and kernel.
  - Mitigation: one canonical pack format spec and converter tests.
- **Performance illusion risk (medium)**: gains from fallback to native path instead of FlagGems.
  - Mitigation: backend-tag evidence in profiler logs must be mandatory.
- **Compile fragility risk (medium)**: Triton CPU backend codegen edge cases on BF16/int kernels.
  - Mitigation: keep per-op fallback and CI smoke matrix.

## 5) Acceptance criteria (ship gate)

- Correctness:
  - [ ] int8 path: max error + relative error within target per op and per model stage.
  - [ ] int4 path: top-k/token-level output drift within agreed tolerance band.
- Path purity:
  - [ ] targeted quantized ops run on FlagGems `_arm` kernels (with proof logs), not silent ATen fallback.
- Performance:
  - [ ] Qwen3-0.6B and Qwen2-7B decode benchmarks show stable improvement or neutral with tighter CPU usage.
- Operability:
  - [ ] single env switch to disable quantized fastpaths for rollback.

## 6) Recommended immediate next sprint

1. Lock P0 evidence pipeline and baseline snapshots.
2. Deliver P1 int8 `_arm` `mm/addmm` with strict shape bucket support.
3. Start P2 int4 with decode-only `M=1` fastpath first, then generalize.
