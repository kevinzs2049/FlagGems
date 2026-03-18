# FlagTree × FlagGems ARM64 CPU 后端集成总结报告

**平台**：CIX P1 CD8180（12 核 ARMv9-A，SVE2 + i8mm + BF16，2.6 GHz 大核）
**软件栈**：triton-cpu `96fc8730`（arm64-dev）+ FlagTree `triton_v3.3.x_arm64` + FlagGems
**日期**：2026-03-05

---

## 1. 背景与目标

FlagGems 是基于 Triton 的算子库，原生支持 CUDA/NPU。本次集成目标是在 ARM64 CPU
平台上，通过 **FlagTree**（定制化 Triton Python 分发层）+ **triton-cpu**（LLVM/SVE2
C++ 后端）的组合，使 FlagGems 的 ARM 算子路径能够正确编译并高效执行，从而加速
BF16/INT8 LLM 推理。

---

## 2. 架构分层

```
FlagGems
│  @triton.jit 算子实现 + torch.library 注册
│  src/flag_gems/runtime/backend/_arm/ops/  （52 个 ARM 专属算子文件）
│
▼  import triton / @triton.jit
FlagTree
│  python/triton/runtime/jit.py      自定义 JIT 流程（兼容 GPU autotune key）
│  python/triton/runtime/build.py    ARM march 修复 + GCC 汇编兼容清理
│  third_party/cpu/backend/
│    compiler.py                     ARM cpu_arch / min_dot_size / SVE2 pass
│    driver.py                       ARM 平台驱动
│  third_party/cpu/language/cpu/     libdevice 数学库（对标 nvidia libdevice）
│  python/triton/backends/cpu        symlink → third_party/cpu/backend
│
▼  调用 libtriton.so
triton-cpu
│  libtriton.so（MLIR → LLVM → AArch64 汇编）
│  ConvertDotToSVE2I8MM LLVM pass（SMMLA 指令生成）
│
▼
Hardware: ARMv9-A SVE2 (128-bit), i8mm, BF16
```

---

## 3. 关键技术问题与修复

### 3.1 FlagTree 侧改动（仓库：flagtree，分支：triton_v3.3.x_arm64）

#### 问题 1：`cpu_arch` 误报为 `x86_64`

LLVM 的 `get_cpu_tripple()` 在 ARM 机器上返回 `x86_64-unknown-linux-gnu`（LLVM
库的编译宿主机三元组，而非运行机器）。`compiler.py` 用它来决定是否启用 SVE2 pass，
导致 ARM 上 SVE2 pass 永远不被启用。

```python
# 修复前（错误）：
cpu_arch = llvm.get_cpu_tripple().split("-")[0]  # → "x86_64" on ARM!

# 修复后：
import platform
cpu_arch = platform.machine()  # → "aarch64"
```

#### 问题 2：`min_dot_size` 拒绝 M=1 GEMV

默认 `min_dot_size = (16, 16, 16)` 使得 M=1 的 `tl.dot()` 调用（decode 阶段矩阵向量
乘）被强制走标量 fallback，无法生成向量化代码。

```python
# 修复：
def min_dot_size(target):
    return lambda lhsType, rhsType: (1, 4, 4)
```

效果：FlagGems `mm_m1_transposed_rhs_kernel` 得以编译，decode GEMV 速度提升 2x。

#### 问题 3：SVE2 i8mm pass 未启用

```python
# 修复（compiler.py make_tttcir()）：
if cpu_arch == "aarch64":
    pm.add_convert_dot_to_sve2_i8mm_pass()
```

效果：INT8 prefill 从标量 fallback（~19 GOPS）提升至 SVE2 SMMLA（~411 GOPS）。

#### 问题 4：`-march=armv9-a` 未正确传递

GCC 12 的 `-mcpu=native` 不可靠地启用 SVE 扩展；LLVM 生成的汇编含有 `.file`/`.loc`/
`.cfi_` 指令，GCC 汇编器不兼容。

```python
# 修复（build.py）：
if platform.machine() == "aarch64":
    cc_cmd += ["-march=armv9-a+sve2+i8mm+bf16+fp16", "-msve-vector-bits=128"]
# 同时清理 .file/.loc/.cfi_ 指令
```

#### 问题 5：`backends/cpu` symlink 缺失

`_discover_backends()` 扫描 `triton/backends/` 目录来发现后端。FlagTree 从未被
`pip install`，因此 `backends/cpu` symlink 不存在 → CPU backend 无法被加载。

```bash
# 修复：
ln -sfn flagtree/third_party/cpu/backend  $SITE/backends/cpu
```

#### 问题 6：3 个断链 symlink

`language/extra/{cuda,hip}`、`tools/extra/cuda` 指向不存在的路径
`/home/kevin/FlagTree/`，导致 `import triton` 在部分情况下失败。已删除。

#### 问题 7：jit.py GPU 专属 key 兼容

GPU autotune 配置含有 `num_buffers_warp_spec`、`waves_per_eu` 等 CPU backend
不认识的 key，导致 `AttributeError`。在 jit.py 中加入白名单过滤。

---

### 3.2 FlagGems 侧改动（仓库：FlagGems，分支：origin/flagtree-integration）

#### 3.2.1 `scaled_dot_product_attention` 恢复启用

FlagTree `cpu_arch` 修复前，SDPA 的 autotune 会因非确定性 kernel 选择导致结果不一
致，被列入 `CUSTOMIZED_UNUSED_OPS`。修复后 determinism 已验证（3 次前向结果完全相
同），现已从 UNUSED 列表中移除，正常注册 ARM CPU 上的 SDPA 实现。

#### 3.2.2 MM / AddMM M=1 转置 config 调优

`MM_M1_TRANSPOSED_CONFIG_TABLE` 和 `ADDMM_M1_TRANSPOSED_CONFIG_TABLE` 基于
CIX P1 实测结果更新，统一改为 `BK=64`：

| Shape | 旧 config | 新 config | 提升 |
|-------|-----------|-----------|------|
| qkv/o M=1 K=3584 N=3584 | (32,16) | (4,64) | 1.76x |
| gate/up M=1 K=3584 N=18944 | (16,32) | (4,64) | 1.21x |
| lm_head M=1 K=3584 N=152064 | (16,32) | (4,64) | 1.21x |

`BK=64` 的优势：每次 K 步读取 64×2=128 字节，恰好填满一条 64-byte 缓存行，
内存访问效率最高；相比 `BK=16/32`，减少 K-loop 迭代开销。

#### 3.2.3 cat 算子优化

原始 ARM cat 实现使用 Triton kernel，每元素执行 ndim 次整数除法（UDIV）计算坐标，
加上约 9μs 的固定 kernel 启动开销，在 decode 阶段 KV-cache cat（小 tensor）中
极度低效。

新实现用 `as_strided + copy_` 替代：

```python
offset = 0
for t in tensors:
    torch.as_strided(out, t.shape, out_strides, offset * dim_stride).copy_(t)
    offset += t.shape[dim]
```

| 场景（KV-cache BF16） | 旧 Triton kernel | 新实现 | ATen |
|-----------------------|-----------------|--------|------|
| S=1 | ~600μs | 18μs | 2μs |
| S=100 | ~120μs | 51μs | 21μs |
| S=2048 | — | 141μs | 147μs |

结论：cat 算子仍慢于 ATen（Python dispatch floor ~12μs），**不应加入推理的
FLAGGEMS_INCLUDE 列表**；但相比原 Triton kernel 提升 3-33x，保留为 fallback 实现。

---

## 4. ARM 算子覆盖概况

`src/flag_gems/runtime/backend/_arm/ops/` 共 46 个算子文件（10905 行），
覆盖 LLM 推理主要热点：

| 类别 | 算子 |
|------|------|
| 矩阵乘 | mm, addmm, bmm（含 M=1 转置 RHS fastpath） |
| 量化推理 | quantized_linear_dynamic（INT8 OneDNN/Triton 双路径） |
| 注意力 | attention（scaled_dot_product_attention） |
| 激活函数 | silu, gelu, softmax, log_softmax |
| 归一化 | rsqrt, mean |
| 逐元素 | add, sub, mul, neg, cos, sin, div, pow |
| 归约 | sum, max, min, argmax, all, any |
| 索引/排列 | embedding, index, index_select, gather, scatter, masked_fill, sort, topk |
| 其他 | cat, cumsum, where, full, zeros, ones, arange 等 |

当前从 UNUSED 列表排除的算子：
- **随机数类**（dropout/rand/randn 等）：CPU 设备无 `torch.cpu.default_generators`
- **逐元素乘/加**（mul/add）：decode 小 tensor 场景实测有轻微回退
- `cat`、`cumsum`、`silu`（部分场景 ATen 更快）
- `scaled_dot_product_attention_backward`（训练专用）

---

## 5. 端到端性能

### 5.1 Qwen3-0.6B BF16（transformers，短 prompt 8 tok，20 new tokens）

测试条件：performance governor，N_RUNS=5，30s 冷却间隔，进程间隔测量。

| 模式 | 绑核 | OMP | tok/s | 对比 |
|------|------|-----|-------|------|
| ATen baseline | 8 大核（0,1,6,7,8,9,10,11） | 8 | 5.61 | — |
| ATen baseline | 6 大核（0,1,10,11,6,7） | 6 | 5.86 | +4%（cores 8,9 较慢拖慢 OMP barrier） |
| FlagGems（55 ops） | 6 大核 | 6 | **6.41** | **+9% vs 同等绑核 / +14% vs 8核基线** |

> 128-token prompt：FlagGems -3.9%（prefill BF16→FP32 cast 回退）。

### 5.2 Qwen2-7B BF16（transformers，短 prompt）

| 模式 | 绑核 | OMP | tok/s | 对比 |
|------|------|-----|-------|------|
| ATen baseline | 8 大核 | 8 | 0.66 | — |
| FlagGems | 6 大核 | 6 | **1.69** | **+2.56x** |

加速原因：ATen 对大 N 的 BF16 矩阵乘内部转 FP32（约 5 GFLOPS）；FlagGems Triton
M=1 GEMV 直接处理原生 BF16（约 10 GFLOPS），decode 阶段单步快 2x。

### 5.3 INT8 微基准（Qwen2-7B shapes，OMP=8）

| 算子 shape | Triton i8mm | OneDNN/ATen | 对比 |
|-----------|------------|-------------|------|
| prefill M=128 K=3584 N=18944 | **411 GOPS** | 539 GOPS | 0.76x（OneDNN 更快） |
| decode M=1 K=3584 N=18944 | 69 GOPS | 67 GOPS | 1.03x（持平） |
| decode M=1 K=3584 N=152064 | 72 GOPS | — | — |

INT8 e2e 慢于 OneDNN（prefill 差距明显），INT8 推理仍推荐 OneDNN 路径。

---

## 6. 安装验证步骤

```bash
# 1. 确认 FlagTree CPU backend 已正确加载
python -c "
import triton.backends.cpu.compiler as c, inspect
f = inspect.getfile(c)
assert 'flagtree' in f, f'错误：加载的是 {f}'
from triton.backends.cpu.compiler import CPUBackend
from triton.backends.compiler import GPUTarget
b = CPUBackend(GPUTarget('cpu', '', 0))
mds = b.get_codegen_implementation()['min_dot_size'](None, None)
assert mds == (1, 4, 4), f'min_dot_size 错误：{mds}'
print('OK: FlagTree CPU backend，min_dot_size=(1,4,4)')
"

# 2. FlagGems mm 算子正常编译运行
python -c "
import sys; sys.path.insert(0, 'src')
import flag_gems
flag_gems.only_enable(include=['mm'])
import torch
a = torch.randn(1, 2048, dtype=torch.bfloat16)
b = torch.randn(2048, 6144, dtype=torch.bfloat16)
c = torch.mm(a, b)
print('OK: FlagGems mm，shape =', c.shape)
"

# 3. cat 准确性（136 个测试 case）
python -m pytest tests/test_special_ops.py -m cat -q
```

---

## 7. 仓库改动统计

### FlagTree（`triton_v3.3.x_arm64` 分支，4 个提交）

| 提交 | 说明 |
|------|------|
| `d0847ea8` | 初始 CPU 后端骨架（compiler.py 370 行，driver.py 500 行，cmake） |
| `88465296` | jit.py GPU key 兼容 + build.py ARM march 初版 |
| `df34ed30` | ARM64 核心修复：cpu_arch / min_dot_size / SVE2 pass / symlinks |
| `0ac02cba` | language/cpu 补齐（libdevice.py 222 行） |

**净改动**：+1652 行，-16 行，+多个 symlink 修正。

### FlagGems（`origin/flagtree-integration` 分支，关键提交）

| 提交 | 说明 |
|------|------|
| `0d1083f6` | SDPA 恢复启用 + MM/AddMM config 调优 + flagtree_arm64_integration.md |
| `59a9dad4` | cat：Triton kernel → as_strided+copy_ |
| `d1d23a0d` | FlagTree 集成兼容修复（Trial） |
| `19b7e8d8` | addmm/rotary 修复，vLLM patch，基准文档 |
| `7807c69f` | quantized_linear_dynamic（INT8 Triton 路径） |
| `2b61918f` | linear dispatch / RMSNorm fused kernel |
| `92731329` | mm 转置 RHS M=1 fastpath + prepack cache |

ARM ops 目录：46 个算子文件，约 10900 行代码。

---

## 8. 已知限制与后续方向

| 问题 | 说明 |
|------|------|
| prefill BF16→FP32 cast | addmm M>1 路径仍有 BF16→FP32 转换，长 prompt 场景轻微回退 |
| INT8 prefill 慢于 OneDNN | Triton i8mm prefill 411 GOPS vs OneDNN 539 GOPS，差距 ~1.3x |
| cat 仍慢于 ATen | Python dispatch floor ~12μs，无法消除；小 tensor 场景不应启用 |
| BF16 小模型无收益 | Qwen3-0.6B / Qwen3-1.7B ATen 已用原生 BF16 BLAS，FlagGems 无加速 |

**后续优化方向**：
1. B-in-alloca：将 B pack 值也写入 memref，减少 prefill GEMM 寄存器 spill，缩小与 OneDNN 的差距
2. addmm prefill M>1 BF16 直通：避免 BF16→FP32 cast，修复长 prompt 回退
3. venv-flagtree：新虚拟环境从零配置 FlagTree，彻底隔离历史 symlink 遗留问题
