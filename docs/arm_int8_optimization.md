# ARM64 INT8 GEMM 优化记录

**平台**：CIX P1 CD8180（ARM64，12 核，SVE2 + i8mm）
**日期**：2026-03-06
**作者**：FlagGems ARM 后端

---

## 背景

PyTorch 的 `torch.ao.quantization.quantize_dynamic` 接口将 `nn.Linear` 替换为
`quantized::linear_dynamic` 算子，在 CPU 上默认由 OneDNN/ACL 实现。

在 ARM64（CIX P1 CD8180）上测试 Qwen2.5-1.5B / Qwen3-1.7B INT8 推理，OneDNN 性能如下：

| 模型 | 短 prompt decode | 长 prompt（含 prefill）|
|------|----------------|----------------------|
| Qwen2.5-1.5B | 7.88 tok/s | 7.46 tok/s |
| Qwen3-1.7B   | 7.43 tok/s | 6.71 tok/s |

目标：通过 FlagGems + Triton-CPU 实现的 SVE2 i8mm kernel 替换 OneDNN，提升性能。

---

## Phase 1：M 分支修复（block 选择策略）

### 问题

原始实现对所有非 M=1 情形使用统一的 fallback：

```python
else:
    BM, BN, BK = 1, 64, 4   # 即使 M=32 也走此路径，~69 GOPS
    grid = (M, N // BN)
```

这导致 M∈[2..63] 的 prefill 场景无法利用 SVE2 i8mm 指令。

### 修复

按 M 的对齐情况分级选择 block 配置：

```python
if M == 1:
    BM, BN, BK = 1, 64, 4       # decode: ConvertDotGeneric, 63 GOPS
elif M == 2:
    BM, BN, BK = 2, 64, 4       # 2-row ConvertDotGeneric
elif M % 64 == 0:
    BM, BN, BK = 64, 64, 32     # large prefill: SVE2 i8mm dynamic ForOp, 411 GOPS
elif M % 8 == 0:
    BM, BN, BK = 8, 64, 32      # medium prefill: SVE2 i8mm dynamic ForOp
elif M % 4 == 0:
    BM, BN, BK = 4, 64, 32      # small prefill: SVE2 i8mm static path
else:
    BM, BN, BK = 1, 64, 4       # fallback: M=3,5,6,7 等
```

### 依据

SVE2 i8mm（`smmla` 指令）由 Triton-CPU 的 `ConvertDotToSVE2I8MM` LLVM Pass 激活，条件：
- 一般路径：`M==2 OR M%4==0`，`N%4==0`，`K%8==0`
- Dynamic ForOp 路径（更高效）：额外要求 `M%8==0`，`K%16==0`

BK=32（vs BK=4）的优势：
- 减少外层 K-loop 迭代次数（K/32 vs K/4，少 8x 迭代）
- 更大的 tile 使 prefetcher 更高效
- 触发 SVE2 smmla 流水线（4×8×4 矩阵乘）

微基准结果（K=3584, N=18944, OMP=8）：

| M | 修复前 | 修复后 | 提升 |
|---|--------|--------|------|
| 8   | 69 GOPS (BK=4) | 102-128 GOPS (BK=32) | 1.5-1.8x |
| 64  | 69 GOPS (BK=4) | **411 GOPS** (BK=32) | **6x** |
| 128 | 69 GOPS (BK=4) | **435 GOPS** (BK=32) | **6.3x** |

---

## Phase 2：量化/反量化融合内核

### 问题分析

Phase 1 修复后，用 torch.profiler 对 Qwen3-1.7B 进行 profiling（20 tokens，短 prompt）：

| 类别 | OneDNN | FlagGems（unfused）| 差距 |
|------|--------|-------------------|------|
| quantized_linear（GEMM） | 74.7 ms/tok | 122.4 ms/tok | +47.7 ms |
| dispatch_other（buffer/cast）| 14.1 ms/tok | 26.4 ms/tok | +12.3 ms |
| quant_overhead（abs/max/item）| 3.6 ms/tok | 7.2 ms/tok | +3.6 ms |
| **总计** | **106 ms/tok** | **175 ms/tok** | **+69 ms** |

`dispatch_other` 的差距（+12.3 ms/tok）来自每次 `quantized::linear_dynamic` 调用中
额外的 7 个 PyTorch 算子：

```python
# 每个 linear 层调用一次（共 ~197 次/token）
x_q = x_abs_max / 127.0           # div
x_q = x_q.round_()                # round_
x_q = x_q.clamp_(-128, 127)       # clamp_
x_q = x_q.to(torch.int8)          # to(int8)
c   = torch.empty(M, N, int32)    # empty（buffer 分配）
# ... _i8mm_kernel ...
c   = c.to(torch.float32)         # to(float32)
c   = c.mul_(out_scale)           # mul_
```

每个算子约 9μs dispatch 开销，197 层 × 7 算子 × 9μs = **12.4 ms/tok**，与测量吻合。

### 解决方案：`_i8mm_fused_kernel`

将量化（FP32→INT8）和反量化（INT32→FP32）融合进 Triton kernel：

```python
@triton.jit
def _i8mm_fused_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    inv_x_scale,   # 标量 float32: 127.0 / x_abs_max
    out_scale,     # 标量 float32: (x_abs_max / 127.0) * w_scale
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # ... 省略 pid/offset 计算 ...

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        # 1. 加载 FP32 激活，在 kernel 内量化
        a_fp32 = tl.load(a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        a_int8 = tl.minimum(tl.maximum(a_fp32 * inv_x_scale, -128.0), 127.0).to(tl.int8)

        # 2. 加载预转置的 INT8 权重
        b = tl.load(b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

        # 3. INT8 GEMM → INT32 累加（SVE2 smmla 路径自动激活）
        acc += tl.dot(a_int8, b)

    # 4. 反量化：INT32 → FP32，直接写输出
    tl.store(c_ptr + ..., acc.to(tl.float32) * out_scale)
```

Host 侧仍需一次不可避免的全局 max 归约：
```python
x_abs_max   = x2d.abs().max().item()   # 唯一保留的 Python 算子
inv_x_scale = 127.0 / x_abs_max
out_scale   = (x_abs_max / 127.0) * w_scale
```

### 效果

| 类别 | FlagGems unfused | FlagGems **fused** | 节省 |
|------|-----------------|-------------------|------|
| quantized_linear | 122.4 ms/tok | 103.3 ms/tok | -19 ms |
| dispatch_other | 26.4 ms/tok | 14.1 ms/tok | **-12.3 ms** |
| quant_overhead | 7.2 ms/tok | 5.6 ms/tok | -1.6 ms |
| **总计** | **175 ms/tok** | **137 ms/tok** | **-38 ms (-22%)** |

### 正确性验证

对所有生产形状（M=1,2,4,8,64,128；K=2048,3584,11776；N=18944）进行验证，
与 unfused 参考实现的最大相对误差均为 0.0000（数值完全一致）。

---

## Phase 3：权重 Tile 化预布局

### 问题

Phase 2 fused kernel 使用行主序（row-major）权重布局 `[K, N]`，stride_bk = N（如 18944）。
在 prefill（M≥4）场景下，每次 K-tile 迭代需要跨越 N 列，B tile 不连续，L2 缓存命中率低。

llama.cpp KleidiAI 通过预打包（pre-tiling）将权重变为 `[K//BK, N//BN, BK, BN]` 布局，
每个 BK×BN tile 连续存放，显著提升 prefill 的 L2 hit rate（约 60%+ 改善）。

### 解决方案：`_i8mm_fused_tiled_kernel`

```
原始 weight_kn [K, N]:          stride_bk = N  →  L2 miss 频繁
转换后 weight_tiled [K//BK, N//BN, BK, BN]:  每 tile 连续 →  L2 hit
```

转换方式（一次性，存入 weight cache）：
```python
weight_tiled = (
    weight_kn                      # int8 [K, N]
    .reshape(K//BK, BK, N//BN, BN)
    .permute(0, 2, 1, 3)           # → [K//BK, N//BN, BK, BN]
    .contiguous()
)
```

Triton kernel 中 B tile 加载：
```python
# 原行主序（L2 miss）
b = tl.load(b_ptr + offs_k[:, None] * N + offs_n[None, :])

# 新 tiled（连续加载，L2 hit）
b_base = b_ptr + (k * N_TILES + pid_n) * BK * BN
b = tl.load(b_base + tl.arange(0, BK)[:, None] * BN + tl.arange(0, BN)[None, :])
```

路由策略：
- 解码（M=1,2）：保持 `weight_kn` 行主序 + `_i8mm_fused_kernel`（BK=4，LLVM 完全展开）
- 预填充（M≥4）：使用 `weight_tiled` + `_i8mm_fused_tiled_kernel`（BK=32，SVE2 i8mm）

额外内存：约 1x 权重大小（Qwen3-1.7B +~1.7 GB，Qwen2.5-1.5B +~1.6 GB），一次性开销。

### 微基准结果（K=3584, N=18944, OMP=8）

| M | 行主序（GOPS） | Tile 化（GOPS） | 提升 |
|---|-------------|--------------|------|
| 4  | 64.3 | 58.8 | -8%（小 M 未受益）|
| 8  | 116.3 | 124.6 | +7% |
| 64 | 254.7 | 263.4 | +3.4% |
| 128 | 208.6 | 232.7 | +11.5% |

---

## E2E 性能结果

测试环境：OMP=8，taskset 绑定 8 大核（cpu0,1,6,7,8,9,10,11），performance governor，N_RUNS=3

### Qwen2.5-1.5B-Instruct INT8

| 引擎 | 短 prompt（8 tok→20）| 长 prompt（84 tok→50）|
|------|---------------------|----------------------|
| OneDNN（基线）| **7.88** tok/s | **7.46** tok/s |
| FlagGems fused（Phase 2）| 6.82 tok/s (0.87x) | 5.38 tok/s (0.72x) |
| FlagGems tiled（Phase 3）| **7.25 tok/s (0.92x)** | **5.66 tok/s (0.76x)** |

### Qwen3-1.7B INT8

| 引擎 | 短 prompt（8 tok→20）| 长 prompt（84 tok→50）|
|------|---------------------|----------------------|
| OneDNN（基线）| **7.43** tok/s | **6.71** tok/s |
| FlagGems fused（Phase 2）| 6.16 tok/s (0.83x) | 4.70 tok/s (0.70x) |
| FlagGems tiled（Phase 3）| **6.55 tok/s (0.88x)** | **4.73 tok/s (0.70x)** |

Phase 3 tiling 相比 Phase 2 fused 的提升：短 prompt +5~6%，长 prompt +0.6~5%。
累计优化链：Phase 1（M 分支）→ Phase 2（融合）→ Phase 3（tile 化），总体达 0.88~0.92x OneDNN。

---

## 剩余差距分析

### decode（短 prompt，M=1）差距：~0.88-0.92x

| 原因 | 估算损失 |
|------|---------|
| Triton i8mm GEMM 内核慢于 OneDNN/ACL | ~15-20 ms/tok |
| abs/max/item dispatch（197 层 × 3 算子 × ~9μs）| ~5 ms/tok |
| 合计 | ~20-25 ms/tok |

M=1 GEMV 是纯内存带宽瓶颈（需读全部权重）。Triton 63 GOPS vs OneDNN 67 GOPS，
差距不大但 e2e 因 OMP 调度开销放大。

### prefill（长 prompt，M=84）差距：~0.70-0.76x

M=84 触发 `BM=4, BK=32` 静态路径（57-73 GOPS），未能进入更高效的 `BM=8` Dynamic ForOp
路径（100-128 GOPS），因为 84 % 8 = 4 ≠ 0。OneDNN/ACL 在中等 M 的 GEMM 上优化更充分。

---

## 后续优化方向（TODO）

### 短期

1. **M=84 prefill padding**：对非 M%8==0 的形状，考虑 padding 到 M%8==0 再调用 BM=8 路径，
   以利用 Dynamic ForOp（目前 84%8=4，走 BM=4 静态路径）

2. **abs/max Triton 化**：用小 Triton kernel 替代 `x2d.abs().max().item()`，
   减少 ~5 ms/tok 的 dispatch 开销（197 层 × 3 × 9μs）

### 中期

3. **覆盖 `aten::_int_mm` CPU dispatch**：
   - torchao `Int8DynamicActivationInt8WeightConfig` 走此接口
   - 当前 ARM64 上 `_int_mm` 是标量实现（1.9 GOPS），比 Triton 慢 33x
   - 实现后可覆盖 torchao 所有量化模型，适配现代量化生态
   - 新建 `_arm/ops/int_mm.py`，注册 `aten::_int_mm` CPU dispatch

### 长期

4. **per-block 激活量化**：对齐 llama.cpp Q8_0（per-32-elements 量化），
   精度更高且可消除全局 abs/max 归约

5. **vLLM torchao 路径**：待 `_int_mm` 覆盖完成后，评估 vLLM CPU INT8 路径

---

## 参考

- Triton-CPU SVE2 i8mm Pass: `triton-cpu/third_party/cpu/lib/Conversion/TritonCPUToLLVM/DotOpToSVE2I8MMConversion.cpp`
- FlagGems ARM 算子: `src/flag_gems/runtime/backend/_arm/ops/quantized_linear_dynamic.py`
- 微基准脚本: `/tmp/test_fused_kernel_correctness.py`, `/tmp/bench_int8_m_sweep.py`
- E2E 测试脚本: `/tmp/quantize_and_bench_small.py`, `/tmp/profile_int8_gap.py`
