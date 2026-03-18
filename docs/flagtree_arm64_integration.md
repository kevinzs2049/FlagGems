# FlagTree ARM64 CPU Backend Integration

This document describes how FlagTree integrates with FlagGems on ARM64 platforms,
the toolchain requirements, and the rationale for each integration layer.

## Architecture Overview

```
FlagGems (operator library)
    │  @triton.jit kernels + torch.library registrations
    ▼
FlagTree (Python distribution layer)
    │  Customized runtime/jit.py, runtime/build.py
    │  third_party/cpu/backend/compiler.py  ← ARM-specific compiler config
    │  third_party/cpu/backend/driver.py    ← ARM platform driver
    ▼
triton-cpu (C++ extension layer)
    │  libtriton.so  ← MLIR/LLVM codegen, SVE2 i8mm lowering pass
    ▼
Hardware: ARMv9-A (e.g. CIX P1 CD8180 with SVE2 + i8mm + bf16)
```

Each layer has a distinct responsibility:

| Layer | Responsibility | Does NOT handle |
|-------|---------------|-----------------|
| **triton-cpu** | MLIR → LLVM → AArch64 assembly; SVE2 i8mm lowering pass | Compiler flags, min_dot_size, Python JIT flow |
| **FlagTree** | ARM compiler flags (`-march`); `min_dot_size`; JIT caching | LLVM passes, assembly generation |
| **FlagGems** | Operator implementations; torch.library dispatch registration | Triton compilation details |

## Toolchain Requirements

### 1. triton-cpu

Source: <https://github.com/triton-lang/triton-cpu>

Required branch/commit: `arm64-dev` (commit `96fc8730` or later).
Key capability: SVE2 i8mm LLVM lowering pass (`ConvertDotToSVE2I8MM`).

Build (one-time, takes several hours):
```bash
cd /path/to/triton-cpu
pip install ninja cmake pybind11
python setup.py develop
# After build, copy libtriton.so to site-packages:
cp python/build/cmake.linux-aarch64-cpython-3.11/triton/_C/libtriton.so \
   $(python -c "import triton; import os; print(os.path.dirname(triton.__file__))")/_C/libtriton.so
```

Or install the pre-built wheel if available:
```bash
pip install triton-cpu  # installs libtriton.so only; Python layer will be overridden by FlagTree
```

### 2. FlagTree

Source: <https://github.com/FlagOpen/FlagTree>

Required branch: `triton_v3.3.x_arm64` (or later ARM64-capable branch).

FlagTree is **not pip-installed** in the traditional sense. Instead its Python files
are applied as symlink overrides on top of the triton-cpu Python installation:

```bash
SITE=$(python -c "import triton; import os; print(os.path.dirname(triton.__file__))")

# 1. FlagTree JIT (custom compilation pipeline)
ln -sf /path/to/flagtree/python/triton/runtime/jit.py   $SITE/runtime/jit.py

# 2. FlagTree build.py (ARM march fix)
ln -sf /path/to/flagtree/python/triton/runtime/build.py $SITE/runtime/build.py

# 3. FlagTree CPU backend (THE KEY: compiler.py with min_dot_size + SVE2)
ln -sfn /path/to/flagtree/third_party/cpu/backend       $SITE/backends/cpu

# 4. CPU math library (libdevice equivalents for CPU)
ln -sfn /path/to/flagtree/third_party/cpu/language/cpu  $SITE/language/extra/cpu
```

### 3. FlagGems

```bash
cd /path/to/FlagGems
pip install -e .
```

## Key Fixes in FlagTree for ARM64

### compiler.py — `cpu_arch` detection

**Problem**: `llvm.get_cpu_tripple()` returns `x86_64-unknown-linux-gnu` even on ARM64
hardware (the LLVM library was cross-compiled and reports its build host).

**Fix**:
```python
# Wrong (upstream triton-cpu):
cpu_arch = llvm.get_cpu_tripple().split("-")[0]  # → "x86_64" on ARM!

# Correct (FlagTree):
import platform
cpu_arch = platform.machine()  # → "aarch64"
```

### compiler.py — `min_dot_size`

**Problem**: The default `min_dot_size = (16, 16, 16)` rejects M=1 `tl.dot()` calls,
forcing decode-phase GEMV (matrix-vector) ops through a slow scalar fallback path.

**Fix**:
```python
# FlagTree compiler.py:
def get_codegen_implementation(self):
    ...
    return {
        "min_dot_size": lambda *_: (1, 4, 4),  # allow M=1 decode GEMV
        ...
    }
```

This enables FlagGems' `mm_m1_transposed_rhs_kernel` to be compiled and run for
decode-phase projections (gate/up/down/qkv/o), giving 2x speedup over ATen on
large-vocabulary models such as Qwen2-7B BF16.

### compiler.py — SVE2 i8mm + BF16 dot passes

**Problem**: SVE2 i8mm and BF16 dot LLVM lowering passes are not enabled by default
in triton-cpu's `make_tttcir()` pipeline on ARM.

**Fix** (in `make_tttcir()` after the BF16 dot product block):
```python
if cpu_arch == "aarch64":
    pm.add_convert_dot_to_sve2_i8mm_pass()
```

This enables the `ConvertDotToSVE2I8MM` pass which lowers `tl.dot` on `int8` inputs
to the `SMMLA` instruction (8-way outer-product per cycle), achieving ~400 GOPS for
INT8 prefill on CIX P1 (vs ~19 GOPS for the scalar fallback).

### build.py — ARM march flag

**Problem**: `-mcpu=native` does not reliably enable SVE2 on GCC 12 / Clang 15 for
the kernel JIT compilation step.

**Fix**:
```python
# FlagTree build.py (ARM64 branch):
if platform.machine() == "aarch64":
    march = "-march=armv9-a+sve2+i8mm+bf16+fp16 -msve-vector-bits=128"
else:
    march = "-march=native"
```

## Verifying the Integration

```bash
python - <<'EOF'
import triton.backends.cpu.compiler as c, inspect

# 1. Confirm FlagTree compiler is active (not upstream triton-cpu)
f = inspect.getfile(c)
print("compiler:", f)
assert "flagtree" in f, f"WRONG backend loaded: {f}"

# 2. Confirm min_dot_size=(1,4,4)
from triton.backends.cpu.compiler import CPUBackend
from triton.backends.compiler import GPUTarget
b = CPUBackend(GPUTarget("cpu", "", 0))
fn = b.get_codegen_implementation()
mds = fn["min_dot_size"](None, None)
print("min_dot_size:", mds)
assert mds == (1, 4, 4), f"WRONG min_dot_size: {mds}"

print("OK: FlagTree CPU backend active")
EOF

# 3. FlagGems mm operator compiles and runs via FlagTree
python - <<'EOF'
import sys; sys.path.insert(0, "src")
import flag_gems
flag_gems.only_enable(include=["mm"])
import torch
a = torch.randn(1, 2048, dtype=torch.bfloat16)
b = torch.randn(2048, 6144, dtype=torch.bfloat16)
c = torch.mm(a, b)
ref = torch.mm(a.float(), b.float()).bfloat16()
assert torch.allclose(c, ref, atol=0.1), "mm result mismatch"
print("OK: FlagGems mm via FlagTree, shape", c.shape)
EOF
```

## Performance Impact

Measured on CIX P1 CD8180 (12-core ARMv9-A, performance governor).

### Qwen3-0.6B BF16 (transformers, short prompt 8 tok, 20 new tokens)

| Mode | Cores | OMP | tok/s | vs baseline |
|------|-------|-----|-------|------------|
| ATen baseline | 8 big | 8 | 5.61 | — |
| ATen baseline | 6 big | 6 | 5.86 | +4% (cores 8,9 are slower) |
| FlagGems (55 ops) | 6 big | 6 | **6.41** | **+9% vs 6-core / +14% vs 8-core** |

> Note: FlagGems decode speedup (+9%) comes primarily from the M=1 transposed GEMV
> kernel. Long-prompt (128-tok) shows slight regression (-4%) due to the BF16→FP32
> cast in addmm for M>1 prefill shapes.

### Qwen2-7B BF16 (transformers, short prompt)

| Mode | Cores | OMP | tok/s | vs baseline |
|------|-------|-----|-------|------------|
| ATen baseline | 8 big | 8 | 0.66 | — |
| FlagGems (mm/addmm/silu/…) | 6 big | 6 | **1.69** | **+2.56x** |

ATen internally converts BF16 → FP32 for large N projections (gate/up N=18944),
limiting it to ~5 GFLOPS. FlagGems' native BF16 GEMV kernel achieves ~10 GFLOPS.

## FlagGems ARM Operator Coverage

FlagGems ARM operators are in `src/flag_gems/runtime/backend/_arm/ops/`.
The ARM backend overrides generic ops (in `src/flag_gems/ops/`) for CPU-specific
implementations (e.g. `cat` uses `as_strided+copy_` instead of a Triton kernel;
`mm` uses M=1 transposed-RHS fastpath for decode GEMV).

Operators excluded on ARM (see `CUSTOMIZED_UNUSED_OPS` in `_arm/__init__.py`):
- Random number ops (`rand`, `randn`, `dropout`, etc.) — require `torch.cpu.default_generators`
- `scaled_dot_product_attention_backward` — training-only
- `mul`/`add` elementwise — profiling shows regression on decode for small tensors

## See Also

- `docs/arm64_int8_int4_migration_checklist.md` — INT8 quantization on ARM64
- `src/flag_gems/runtime/backend/_arm/ops/mm.py` — M=1 GEMV decode fastpath
- FlagTree source: `third_party/cpu/backend/compiler.py`
- triton-cpu source: `third_party/cpu/lib/Conversion/TritonCPUToLLVM/DotOpToSVE2I8MMConversion.cpp`
