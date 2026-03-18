# Qwen3-0.6B BF16 推理：FlagGems 基准测试报告

**平台**: CIX P1 CD8180 (ARM64, 12 cores)
**模型**: Qwen3-0.6B BF16 (`Qwen/Qwen3-0___6B`)
**测试环境**: performance governor，N_RUNS=5，30s cooldown，顺序测试（baseline 和 FlagGems 分开进程）
**基准脚本**: `examples/model_qwen3_0_6b_e2e_bench.py`

---

## 性能基准（2026-02-27）

### 短 prompt（8 tok in，20 tok out）

| 推理引擎 | 绑核 | OMP | tok/s | vs 同等绑核基线 | vs 8核基线 |
|---------|------|-----|-------|--------------|----------|
| ATen 基线 | 8大核 (0,1,6,7,8,9,10,11) | 8 | 5.61 | — | — |
| ATen 基线 | 6大核 (0,1,10,11,6,7) | 6 | 5.86 | — | +4.5% |
| FlagGems only_enable | 6大核 (0,1,10,11,6,7) | 6 | **6.41** | **+9.4%** | **+14%** |

> ATen 6核 > ATen 8核 原因：cpu8/cpu9 频率仅 2200MHz，OMP barrier 等待拖慢整体；6大核去掉这两个慢核反而更快。

### 长 prompt（126 tok in，50 tok out）

| 推理引擎 | 绑核 | OMP | tok/s | vs 8核基线 |
|---------|------|-----|-------|----------|
| ATen 基线 | 8大核 | 8 | **4.86** | — |
| FlagGems only_enable | 6大核 | 6 | 4.67 | **-3.9%** |

> 长 prompt 轻微回退原因：prefill 阶段 addmm M>1 走 BF16→FP32 强制转换（LLVM masked_load crash 的临时修复），
> 且少用 2 个核心。decode 步骤本身仍有 ~+9% 提速，但被 prefill 代价拉低。

---

## `only_enable` 算子清单与收益分析

### ✅ 高价值算子（走 FlagGems，明确收益）

| 算子 | 在模型里的用途 | 收益原因 |
|------|--------------|--------|
| `mm` / `mm_out` | Decode Linear（M=1 GEMV） | 原生 BF16，避开 ATen BF16→FP32 转换，带宽效率翻倍 |
| `addmm` / `addmm_out` | Prefill Linear（M>1 GEMM） | Triton 向量化，但 M>1 有 BF16→FP32 cast（prefill 轻微回退） |
| `bmm` | Attention Q×K^T 和 Attn×V | 批量矩阵乘，Triton 优化路径 |
| `scaled_dot_product_attention` | 完整 Self-Attention | 融合 QKV→输出，减少内存往返 |
| `silu` / `silu_` | MLP SwiGLU 激活 | 融合减少内存往返 |
| `softmax` | Attention softmax（独立路径） | 融合 |
| `rsqrt` / `rsqrt_` | RMSNorm：`rsqrt(mean + eps)` | 频率中等，Triton 略快 |
| `mean` / `mean_dim` | RMSNorm：`.mean(-1, keepdim=True)` | 同上 |

---

### ~ 中性算子（走 FlagGems，各 <1%，合计约 -6%）

| 算子组 | 用途 | 单独影响 |
|-------|------|---------|
| `neg` / `cos` / `sin` | RoPE 旋转位置编码 | ~-4%（M=1 Triton ~15μs vs ATen ~3μs，但 decode 需要） |
| `gelu` / `gelu_` | （Qwen3 不使用，备用） | 中性 |
| `sub` / `sub_` | 残差减法（少量） | ~-1% |
| `pow_tensor_scalar` 等 | RMSNorm：`x.pow(2)` | ~-1% |
| `log_softmax` | 少量调用 | 中性 |
| `sum` / `sum_dim` / `max` / `min` / `argmax` | 归约类 | 中性 |
| `embedding` | Token 查表（prefill 主导，decode 1次） | 中性 |
| `gather` / `index` / `index_select` / `scatter` | 采样、索引 | 低频，中性 |
| `sort` / `sort_stable` / `topk` / `multinomial` | Top-k 采样 | 低频，中性 |
| `where_self_out` / 除法类 | 条件选择、除法 | 低频，中性 |
| `masked_fill` / `masked_fill_` | Attention mask 填充 | 低频，中性 |

---

### ❌ 排除算子（启用后性能回退，保持 ATen）

| 算子 | 用途 | 排除原因 | 实测影响 |
|------|------|---------|---------|
| `mul` / `mul_` | RMSNorm weight 缩放、SwiGLU `silu(gate)×up`、RoPE | `weight[1024]×hidden[1,1,1024]` 广播 shape 走 `@pointwise_dynamic`；Triton ~73μs vs ATen ~3μs；252+ calls/token | **-28% e2e 回退** |
| `add` / `add_` | 每层 2 个残差连接（28层×2=56次） | `[1,1,1024]` 极小 tensor；56 calls × ~30μs 纯启动开销 | **-8% e2e 回退** |
| `patch_qwen3_rmsnorm()` | Fused RMSNorm（2-pass Triton） | M=1 decode 比 ATen 5-op 分解（pow→mean→rsqrt→mul）更慢 | 负收益 |

---

## 算子组合消融实验（OMP=6, taskset 6大核）

| 配置 | tok/s | 备注 |
|------|-------|------|
| v1 核心 12 ops（GEMM/SDPA/silu/rsqrt/mean） | **6.51** | 最纯粹高价值 |
| + 一元算子（neg/cos/sin/gelu） | 6.26 | -4%（RoPE 小 tensor 开销） |
| + sub | 6.45 | -1% |
| + pow | 6.47 | -1% |
| + 归约+杂项 | 6.45 | -1% |
| **+ 全部扩展（当前配置）** | **6.09** | -6%（含 neg/cos/sin 等中性开销） |
| + add | 6.00 | -8% ← 排除 |
| + mul | 4.70 | **-28%** ← 排除 |
| **最终: 当前配置 N_RUNS=5 实测** | **6.41** | N_RUNS=3 消融 vs N_RUNS=5 正式测 差异正常 |

---

## 每步 Decode 算子执行路径

Qwen3-0.6B 架构：hidden=1024，28层，GQA

```
token_ids
  → [embedding] ✅ FlagGems
  → hidden [1×1024]

每层（×28）:
  ┌─ RMSNorm (input_layernorm)
  │    pow ✅ → mean ✅ → rsqrt ✅
  │    hidden × norm_weight  ← mul ❌ ATen（排除）
  │
  ├─ QKV Linear
  │    [1×1024] × [1024×1024]  ← mm ✅ FlagGems（核心）
  │
  ├─ RoPE
  │    cos/sin ✅（~-4% 但保留）
  │    rotate_half 里的 neg/mul/cat ← neg ✅, mul ❌ ATen, cat ❌ ATen
  │
  ├─ SDPA ✅ FlagGems
  │
  ├─ O proj  ← mm ✅ FlagGems
  │
  ├─ 残差 add ❌ ATen（排除）
  │
  ├─ RMSNorm (post_attention_layernorm)
  │    pow ✅ → mean ✅ → rsqrt ✅
  │    hidden × norm_weight  ← mul ❌ ATen
  │
  ├─ MLP (SwiGLU)
  │    gate_proj  ← mm ✅
  │    up_proj    ← mm ✅
  │    silu(gate) ✅ → silu × up  ← mul ❌ ATen（排除）
  │    down_proj  ← mm ✅
  │
  └─ 残差 add ❌ ATen（排除）

lm_head  ← mm ✅ FlagGems
  → argmax ✅ → 输出 token
```

---

## 结论与建议

| 场景 | 推荐配置 | 相比 8核 ATen 基线 |
|------|---------|-----------------|
| Decode 密集（短 prompt） | FlagGems OMP=6，taskset 6大核 | **+14%** |
| 长 prompt（prefill 占比大） | ATen OMP=8，taskset 8大核 | 基线（FlagGems -3.9%） |

**关键经验**：
1. `mul`/`add` 必须排除——小 tensor Triton 启动开销远超计算量
2. 不能用 `flag_gems.enable()`（全量）——`arange`/`copy_` 等工具类 op 干扰 transformers 生成循环
3. Baseline 和 FlagGems **必须分开进程**、不同 taskset，否则绑核竞争导致结果失真
4. CPU governor 必须设为 `performance`，schedutil 下 CPU 不在最高频，数据不可信

---

## 使用方式

```bash
# 设置 performance governor（需 root）
for cpu in $(seq 0 11); do
  echo "performance" > /sys/devices/system/cpu/cpu${cpu}/cpufreq/scaling_governor
done

# 1. Baseline（8大核，OMP=8）
taskset -c 0,1,6,7,8,9,10,11 python examples/model_qwen3_0_6b_e2e_bench.py \
  --baseline-only --baseline-omp 8

# 冷却 30s

# 2. FlagGems（6大核，OMP=6）
taskset -c 0,1,10,11,6,7 python examples/model_qwen3_0_6b_e2e_bench.py \
  --skip-baseline --flaggems-omp 6
```
