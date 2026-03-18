# Qwen2-7B BF16/FP32 推理：FlagGems 算子启用说明

**平台**: CIX P1 CD8180 (ARM64, 12 cores)
**测试环境**: performance governor, N_RUNS=3, 60s cooldown, sequential
**激活方式**: `flag_gems.only_enable(include=FLAGGEMS_INCLUDE)`

---

## 性能基准（2026-02-28）

### 短 prompt（8 tok in，20 tok out）

| 模型格式 | 推理引擎 | 绑核 | OMP | tok/s | vs ATen基线 |
|---------|---------|------|-----|-------|------------|
| INT8    | ATen / OneDNN | 8大核 | 8 | **3.39** | — |
| INT8    | FlagGems Triton i8mm | 6大核 | 6 | 2.46 | -27% |
| BF16    | ATen 基线 | 8大核 | 8 | 0.66 | — |
| BF16    | FlagGems only_enable | 6大核 | 6 | **1.69** | **+156%** |
| FP32    | ATen 基线 | 8大核 | 8 | 0.33 | — |
| FP32    | FlagGems only_enable | 6大核 | 6 | **1.01** | **+206%** |

### 长 prompt（126 tok in，50 tok out）

| 模型格式 | 推理引擎 | 绑核 | OMP | tok/s | vs ATen基线 |
|---------|---------|------|-----|-------|------------|
| INT8    | ATen / OneDNN | 8大核 | 8 | **3.07** | — |
| INT8    | FlagGems Triton i8mm | 6大核 | 6 | 1.38 | -55% |
| BF16    | ATen 基线 | 8大核 | 8 | 0.62 | — |
| BF16    | FlagGems only_enable | 6大核 | 6 | **0.94** | **+52%** |
| FP32    | ATen 基线 | 8大核 | 8 | 0.32 | — |
| FP32    | FlagGems only_enable | 6大核 | 6 | **0.72** | **+125%** |

---

## `only_enable` 算子清单

### ✅ 走 FlagGems 的算子

| 算子 | 在模型里的用途 |
|------|--------------|
| `mm` / `mm_out` | Linear 层前向（decode 阶段，M=1 GEMV） |
| `addmm` / `addmm_out` | Linear 层前向（prefill 阶段，M>1 GEMM） |
| `bmm` | Attention 中 Q×K^T 和 Attn×V |
| `scaled_dot_product_attention` | 整个 Self-Attention（Q/K/V→输出，含 softmax） |
| `silu` / `silu_` | MLP 的 SwiGLU 激活：`silu(gate_proj(x))` |
| `gelu` / `gelu_` | （Qwen2 不使用，备用） |
| `softmax` / `log_softmax` | Attention softmax（SDPA 内已含，独立路径备用） |
| `embedding` | 第一层 token embedding 查表 |
| `pow_tensor_scalar` / `pow_scalar` | RMSNorm：`x.pow(2)` |
| `mean` / `mean_dim` | RMSNorm：`.mean(-1, keepdim=True)` |
| `rsqrt` / `rsqrt_` | RMSNorm：`torch.rsqrt(mean + eps)` |
| `cos` / `cos_` | RoPE 旋转位置编码 |
| `sin` / `sin_` | RoPE 旋转位置编码 |
| `sub` / `sub_` | 通用减法 |
| `neg` / `neg_` | 取负（RoPE rotate_half 中） |
| `sum` / `sum_dim` | 归约求和 |
| `max` / `max_dim` / `min` | 归约 max/min |
| `argmax` | Greedy decode 选 top-1 token |
| `gather` | 采样阶段 token 选取 |
| `index` / `index_select` | 索引操作 |
| `embedding` | token 查表 |
| `scatter` / `scatter_` | 散射写入 |
| `masked_fill` / `masked_fill_` | Attention mask 填充 |
| `sort` / `sort_stable` / `topk` | Top-k 采样 |
| `multinomial` | 随机采样 |
| `where_self_out` | 条件选择 |
| `true_divide` / `floor_divide` / `remainder` | 除法类算子 |

---

### ❌ 走 ATen（未启用 FlagGems）的算子

| 算子 | 在模型里的用途 | 未启用原因 |
|------|--------------|----------|
| `mul` / `mul_` | RMSNorm weight 缩放、SwiGLU 的 `silu(gate)×up`、RoPE rotate | Decode 小 shape Triton 启动开销 > 计算收益，**实测 -19%** |
| `add` / `add_` | 每层两个残差连接（28层×2=56次） | 同上，**实测 -7.5%** |
| `cat` | KV cache 拼接（每步追加 K/V） | 未包含 |
| `view` / `reshape` | 注意力 head reshape，QKV 拆分 | 纯元数据操作，无计算收益 |
| `transpose` / `permute` | 注意力 head 维度转置 | 同上 |
| `expand` / `repeat` | GQA 中 KV head 扩展（4头→28头） | 未包含 |
| `copy_` / `clone` | 各种 tensor 复制 | 未包含 |
| `rms_norm` / `layer_norm` | HuggingFace fused RMSNorm 路径 | 拆成子算子各自优化 |

---

## 每步 Decode 算子执行路径

```
token_ids
  → [embedding] ✅ FlagGems
  → hidden [1×3584]

每层（×28）:
  ┌─ RMSNorm
  │    pow ✅ → mean ✅ → rsqrt ✅
  │    hidden × norm_weight  ← mul ❌ ATen
  │
  ├─ QKV Linear
  │    [1×3584] × [3584×3584]  ← mm ✅ FlagGems（核心加速点）
  │
  ├─ RoPE
  │    cos/sin ✅，rotate_half 里的 mul/cat ❌ ATen
  │
  ├─ SDPA ✅ FlagGems（Q/K/V → attention output）
  │
  ├─ O proj  ← mm ✅ FlagGems
  │
  ├─ 残差 add ❌ ATen
  │
  ├─ MLP
  │    gate_proj  ← mm ✅
  │    up_proj    ← mm ✅
  │    silu(gate) ✅ → silu × up  ← mul ❌ ATen
  │    down_proj  ← mm ✅
  │
  └─ 残差 add ❌ ATen

lm_head Linear ← mm ✅ FlagGems
  → argmax ✅ → 输出 token
```

---

## 加速原理

**核心加速来自 `mm`/`addmm`（Linear 层）**

Decode 阶段每步有 **7 个 Linear 层**（q/k/v proj × 3，o proj × 1，gate/up/down proj × 3），
全部是 M=1 的 GEMV（矩阵×向量）。

| | ATen CPU BF16 | FlagGems Triton BF16 |
|--|--------------|---------------------|
| 计算路径 | BF16 → 转 FP32 → BLAS → 转回 BF16 | 原生 BF16，M=1 fastpath |
| 带宽效率 | ~50%（转换浪费） | ~100% |
| 加速比 | 1× | **~2.5×** |

`mul`/`add` 刻意排除：每次调用 Triton 内核都有 ~9μs 启动开销，
而残差 add（shape=[1×3584]）的计算只需 ~1μs，得不偿失。

---

## 使用方式

```python
import flag_gems

FLAGGEMS_INCLUDE = [
    'mm','mm_out','addmm','addmm_out','bmm','scaled_dot_product_attention',
    'silu','silu_','gelu','gelu_','softmax','log_softmax',
    'rsqrt','rsqrt_','mean','mean_dim',
    'neg','neg_','cos','cos_','sin','sin_',
    'sub','sub_',
    'pow_scalar','pow_tensor_scalar','pow_tensor_scalar_',
    'pow_tensor_tensor','pow_tensor_tensor_',
    'true_divide','true_divide_','floor_divide','floor_divide_',
    'remainder','remainder_','div_mode','div_mode_',
    'sum','sum_dim','max','max_dim','min','argmax',
    'embedding','index','index_select','gather',
    'scatter','scatter_','masked_fill','masked_fill_',
    'sort','sort_stable','topk','multinomial','where_self_out',
]

flag_gems.only_enable(include=FLAGGEMS_INCLUDE)
```

完整基准测试脚本：`/tmp/qwen2_bench.py`
