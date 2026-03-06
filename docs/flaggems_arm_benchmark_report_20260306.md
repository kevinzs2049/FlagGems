# FlagGems ARM64 CPU 推理性能报告

**平台**：CIX P1 CD8180（12 核 ARM64，4 大簇 + 1 小簇）
**日期**：2026-03-06
**绑核**：`taskset -c 0,1,6,7,8,9,10,11`（8 个大核）
**CPU 频率**：performance governor（2.6 / 2.5 / 2.3 GHz）
**OMP_NUM_THREADS**：8
**N_RUNS**：3，进程间隔 20s，各模型/引擎独立进程
**FlagGems**：`/home/kevin/FlagGems-rebase/FlagGems`（含 Phase1-4 优化）
**Triton-CPU**：venv-flagtree（commit 96fc8730，SVE2 i8mm，min_dot_size=(1,4,4)，OMP fix）

---

## 一、性能基准结果

### 1.1 测试方案说明

| 指标 | 说明 |
|------|------|
| **短 prompt** | 输入 8 token，生成 20 新 token（decode 主导） |
| **长 prompt** | 输入 84 token，生成 50 新 token（prefill + decode 混合） |
| **llama.cpp pp512** | 512 token prefill 吞吐（tok/s） |
| **llama.cpp tg128** | 128 token decode 吞吐（tok/s） |
| **INT8 格式** | PyTorch `quantize_dynamic`（per-tensor 动态量化）；llama.cpp Q8_0（per-block 量化） |
| **BF16 格式** | Hugging Face AutoModelForCausalLM，`torch_dtype=bfloat16` |

### 1.2 llama.cpp Q8_0 基准（性能天花板参考）

> 版本：llama-bench b5929，8 线程，taskset 0,1,6,7,8,9,10,11

| 模型 | 参数量 | pp512 (tok/s) | tg128 (tok/s) |
|------|--------|--------------|--------------|
| Qwen3-0.6B | 0.6B | 252.73 | 27.66 |
| Qwen3-1.7B | 1.7B | 102.18 | 12.55 |
| Qwen2.5-1.5B | 1.5B | 112.24 | 13.48 |
| Qwen2-7B | 7B | 23.97 | 3.96 |

### 1.3 INT8 动态量化推理对比（PyTorch, tok/s）

| 模型 | 引擎 | 短prompt (8→20) | 长prompt (84→50) | vs llama.cpp tg128 |
|------|------|----------------|----------------|-------------------|
| **Qwen3-0.6B** | ATen OneDNN (基线) | 9.42 ± 0.22 | 8.58 ± 0.07 | 34% of 27.66 |
| | **FlagGems Triton** | 8.23 ± 0.25 | 7.10 ± 0.06 | 30% of 27.66 |
| | *FlagGems vs ATen* | *-12.6%* | *-17.2%* | |
| **Qwen3-1.7B** | ATen OneDNN (基线) | 7.77 ± 0.58 | 7.00 ± 0.07 | 62% of 12.55 |
| | **FlagGems Triton** | 6.38 ± 0.19 | 5.14 ± 0.13 | 51% of 12.55 |
| | *FlagGems vs ATen* | *-17.9%* | *-26.6%* | |
| **Qwen2.5-1.5B** | ATen OneDNN (基线) | 8.34 ± 0.34 | 8.04 ± 0.23 | 62% of 13.48 |
| | **FlagGems Triton** | 7.15 ± 0.03 | 5.98 ± 0.02 | 53% of 13.48 |
| | *FlagGems vs ATen* | *-14.3%* | *-25.6%* | |
| **Qwen2-7B** † | ATen OneDNN (基线) | 3.39 | 3.07 | 86% of 3.96 |
| | **FlagGems Triton** | 2.51 | 1.80 | 63% of 3.96 |
| | *FlagGems vs ATen* | *-25.9%* | *-41.4%* | |

> † Qwen2-7B 数据来自之前测试（performance governor，N_RUNS=3，sequential）。PT 文件重建中，后续补测最新 kernel 数据。

**INT8 结论**：
- FlagGems Triton INT8 在所有模型上均**慢于** ATen OneDNN（-13% 至 -41%）
- 小模型（0.6B-1.7B）decode 差距约 -13% 至 -18%；大模型（7B）差距扩大至 -26%
- ATen OneDNN 使用 NEON SMMLA 指令（up to 539 GOPS prefill），Triton i8mm 实测约 411 GOPS
- 与 llama.cpp Q8_0 的差距更大（PyTorch INT8 约为 llama.cpp 的 30-86%），主要原因：
  1. llama.cpp 使用 per-block(32) 量化，精度和局部性更好
  2. llama.cpp 直接 NEON SMMLA 汇编，无 Triton JIT 开销

### 1.4 BF16 推理对比（PyTorch, tok/s）

| 模型 | 引擎 | 短prompt (8→20) | 长prompt (84→50) | 加速比(短) | 加速比(长) |
|------|------|----------------|----------------|----------|----------|
| **Qwen3-0.6B** | ATen baseline | 5.03 ± 0.31 | 4.50 ± 0.03 | — | — |
| | **FlagGems Triton** | **8.02 ± 0.26** | **6.33 ± 0.06** | **+59%** | **+41%** |
| **Qwen3-1.7B** | ATen baseline | 2.06 ± 0.03 | 1.97 ± 0.01 | — | — |
| | **FlagGems Triton** | **4.79 ± 0.01** | **3.70 ± 0.01** | **+2.32x** | **+1.88x** |
| **Qwen2.5-1.5B** | ATen baseline | 2.25 ± 0.01 | 2.16 ± 0.01 | — | — |
| | **FlagGems Triton** | **5.31 ± 0.02** | **4.11 ± 0.02** | **+2.36x** | **+1.90x** |
| **Qwen2-7B** † | ATen baseline | 0.66 | 0.62 | — | — |
| | **FlagGems Triton** | **1.69** | **0.94** | **+2.56x** | **+1.52x** |

**BF16 结论**：
- FlagGems Triton 在 **所有 BF16 模型上均大幅领先** ATen（+41% 至 +2.56x）
- 加速来源：ATen 在大模型上 BF16 矩阵乘走 FP32 BLAS（带宽浪费 2x），FlagGems Triton GEMV(M=1) 原生处理 BF16，直接利用 DRAM 带宽
- Qwen3-0.6B 加速比（+59%）小于大模型（+2.3x+），因为 ATen 对 0.6B 模型已有部分 native BF16 BLAS 路径（hidden=1024 较小，缓存友好）

### 1.5 性能对比汇总（短 prompt decode，tok/s）

```
模型            INT8-ATen  INT8-FlagGems  BF16-ATen  BF16-FlagGems  llama.cpp-Q8_0
──────────────  ─────────  ─────────────  ─────────  ─────────────  ──────────────
Qwen3-0.6B        9.42         8.23          5.03         8.02           27.66
Qwen3-1.7B        7.77         6.38          2.06         4.79           12.55
Qwen2.5-1.5B      8.34         7.15          2.25         5.31           13.48
Qwen2-7B†         3.39         2.51          0.66         1.69            3.96
```

---

## 二、FlagGems 算子覆盖分析

> 通过 `torch.profiler` 实测 Qwen3-0.6B forward pass dispatch，10 token decode，精确记录每个算子走向。

### 2.1 BF16 路径：`only_enable(include=[...])`

**注册列表**：`["mm", "addmm", "bmm", "softmax", "silu", "gelu", "mean", "rsqrt", "embedding"]`

#### 实际走 FlagGems Triton 的算子

| ATen 算子 | 调用次数 | 覆盖场景 | 实现细节/注意事项 |
|-----------|---------|---------|----------------|
| `aten::mm` | 1970 | 所有无 bias 线性层（qkv, o_proj, gate/up/down, lm_head）| M=1 transposed 特化核；M%8=0 走 BM=8 BK=32 |
| `aten::bmm` | 10 | Qwen3 q_norm/k_norm 或非 SDPA 路径的 batch matmul | 无 M=1 特化，通用 Triton kernel |
| `aten::silu` | 280 | SwiGLU 激活（每层 1 次）| ⚠️ **BF16 须 cast**：BF16→FP32→Triton→copy 回 BF16，额外两次 tensor copy |
| `aten::mean` | 1130 | RMSNorm 方差（每层 2 norm）| ⚠️ **numel < 4096 走 numpy**：所有 decode 形状 [1, 1024-3584] 均触发 numpy 路径，非 Triton |
| `aten::rsqrt` | 1130 | RMSNorm 归一化 | ⚠️ **numel < 4096 且 dtype∈{fp32,fp64} 走 numpy**；BF16 输入时视 upcast 情况而定 |
| `aten::embedding` | 10 | Token embedding lookup | Triton kernel，每次 generate 调用 1 次 |

#### 注册了但从未被调用的算子

| 注册算子 | 实际调用次数 | 未调用原因 |
|---------|------------|-----------|
| `aten::addmm` | 0 | Qwen3/Qwen2 所有 Linear 无 bias，走 `mm` 而非 `addmm`（有 bias 模型会调用） |
| `aten::_softmax` | 0 | SDPA（Flash Attention）是 fused kernel，内部 softmax 不单独暴露为 aten 调用 |
| `aten::gelu` | 0 | Qwen3/Qwen2 使用 SwiGLU（silu），不使用 GELU（Llama/Mistral 等会调用） |

#### 落回 ATen 的高频算子

| ATen 算子 | 调用次数 | 耗时/call | 原因类别 | 说明 |
|-----------|---------|----------|---------|------|
| `aten::scaled_dot_product_attention` | 280 | 165µs | **主动排除（已测试）** | 独立测试显示注册后 -16% regression；PyTorch CPU FlashAttention 已内置优化 |
| `aten::add` | 2280 | 13.9µs | **主动排除（已测试）** | Residual 连接；实测对 decode 有害，排除 |
| `aten::mul` | 3700 | 10.2µs | **主动排除（已测试）** | SwiGLU gate 乘法（gate × up_proj）；实测有害，排除 |
| `aten::div_` | 1130 | 19.9µs | 未评估 | RoPE head normalization（`q /= sqrt(head_dim)`）|
| `aten::neg` | 560 | 8.3µs | 未评估 | RoPE 旋转位置编码（负值翻转部分）|
| `aten::pow` | 1130 | 11.3µs | 未评估 | RMSNorm x² 计算 |
| `aten::sum` | 1130 | 10.0µs | 未评估 | RMSNorm 方差路径（部分 Transformers 版本使用） |
| `aten::cat` | 1150 | 20.1µs | 未评估 | KV-cache concat（每层每 step 2 次）|
| `aten::cos` / `aten::sin` | 10 | 12µs | 未评估 | RoPE 角度，prefill 期间 1 次 |
| `aten::_to_copy` / `aten::to` | 1300+ | 14µs | 基础设施 | dtype/device 转换；部分来自 silu BF16 cast 自身 |
| `aten::linear` | 1970 | 459µs | 包装层 | `nn.Linear.forward` 内部调用 `mm`，FlagGems 在 `mm` 层拦截；两者都出现在 profiler |

### 2.2 INT8 路径：仅注册 `quantized::linear_dynamic`

| 算子 | 走向 | 调用次数 | 说明 |
|------|------|---------|------|
| `quantized::linear_dynamic` | **FlagGems Triton** | 1970 | 所有量化线性层；M=1/2/M%8/M%64 各有专用分支 |
| `aten::abs` | ATen | 3940 | **INT8 量化激活**：每次 linear 前 `x.abs().max()` 求 per-tensor x_scale（2 call/layer）|
| `aten::silu` | ATen | 280 | FP32 SiLU（INT8 模型激活层保持 FP32）|
| `aten::mean` | ATen | 1130 | RMSNorm（FP32，极快，numel=1024 全走 ATen） |
| `aten::rsqrt` | ATen | 1130 | RMSNorm（FP32）|
| `aten::scaled_dot_product_attention` | ATen | 280 | Flash Attention（FP32）|
| `aten::add` | ATen | 2280 | Residual（FP32）|
| `aten::mul` | ATen | 3700 | SwiGLU gate（FP32）|
| 其余约 60 个 aten 算子 | ATen | — | 全部 ATen，INT8 模型非线性部分完全不涉及量化 |

**INT8 算子说明**：PyTorch `quantize_dynamic` 只量化 `nn.Linear`（权重 INT8 存储，激活运行时动态量化）。所有激活函数、归一化、注意力、位置编码均保持 FP32，直接走 ATen 原生实现。FlagGems 仅拦截 `quantized::linear_dynamic` 一个算子。

### 2.3 FlagGems 算子内部 Fallback 条件（实现层）

| 算子 | 触发 Fallback 的条件 | Fallback 方式 | 影响 |
|------|---------------------|-------------|------|
| `mm` | `N < 256 and M ≤ 8 and dtype ∈ {fp32, fp64}` | **numpy.dot** | 小矩阵避免 Triton 启动开销（~9µs） |
| `mean` | `numel < 4096 and is_contiguous()` | **numpy.mean** | **decode 路径全部触发**（[1,1024-3584] < 4096）|
| `rsqrt` | `numel < 4096 and dtype ∈ {fp32, fp64}` | **numpy rsqrt** | decode RMSNorm 归一化标量 [1,1,1] 触发 |
| `softmax` | `device.type == "cpu"`（始终） | **numpy softmax** | CPU 推理时永远走 numpy；decode 实际不触达（SDPA 拦截）|
| `silu` | `dtype == bfloat16` | BF16→FP32 cast 后进 Triton kernel，再 cast 回 | 额外 2 次 tensor copy/cast，对大 tensor（intermediate=3072-18944）开销可观 |

### 2.4 算子路径总览图

```
Transformer Decode 单步（BF16 FlagGems）
─────────────────────────────────────────────────────────────────
Token Embedding         aten::embedding        → FlagGems Triton
                                               ─────────────────
For layer in 0..27:
  Input LayerNorm
    x² 计算             aten::pow              → ATen
    mean(x²)           aten::mean             → FlagGems (numpy, numel<4096)
    rsqrt(var+eps)     aten::rsqrt            → FlagGems (numpy, numel<4096)
    x * scale          aten::mul              → ATen (主动排除)

  QKV Projection       aten::mm               → FlagGems Triton (M=1 transposed)
  RoPE                 aten::cos/sin/neg/div_ → ATen (未评估)
  Attention (SDPA)     aten::scaled_dot_product_attention → ATen (主动排除,-16%)
  O Projection         aten::mm               → FlagGems Triton
  Residual Add         aten::add              → ATen (主动排除)

  Post-Attn LayerNorm  (同上 mean/rsqrt)      → FlagGems (numpy)

  Gate Projection      aten::mm               → FlagGems Triton
  Up Projection        aten::mm               → FlagGems Triton
  SiLU Activation      aten::silu             → FlagGems Triton (BF16 cast)
  Gate × Up            aten::mul              → ATen (主动排除)
  Down Projection      aten::mm               → FlagGems Triton
  Residual Add         aten::add              → ATen (主动排除)

LM Head               aten::mm               → FlagGems Triton
Logits Argmax         aten::argmax           → FlagGems Triton ✓ (2.0x, auto-registered)
─────────────────────────────────────────────────────────────────
```

---

## 三、INT8 路径算子替换实验结果（2026-03-06）

以 Qwen3-1.7B INT8 decode (M=1, FP32 激活) 为基准：

| 步骤 | 算子 | 测试结果 | 结论 |
|------|------|---------|------|
| Step 1 | `rms_norm` (patch_qwen3_rmsnorm) | ATen 20.6µs vs Triton 25.6µs（0.80x） | **REGRESSION** -0.3ms/token，跳过 |
| Step 2 | `silu` (FP32 [1,6144]) | ATen 20µs vs Triton 73µs（0.27x） | **REGRESSION** -3.5ms/token，跳过 |
| Step 3 | `abs_max` (fused kernel) | ATen 4µs vs Triton 27µs（0.15x） | **REGRESSION** -3.2ms/token，跳过 |
| Step 4 | `argmax` [1,151936] | ATen 584µs vs FlagGems 289µs（**2.0x**） | ✓ **集成**，~294µs/token 节省 |

**关键发现**：M=1 decode 形状下，所有小张量 FP32 算子（numel < 100K）ATen NEON 均快于 Triton-CPU：
- Triton 最小启动开销 ~17µs；ATen NEON 对 H=2048 FP32 单遍扫描 ≈ 3-20µs
- 只有 argmax [1,152K] 张量足够大，Triton 并行归约摊销了启动开销

**argmax 已集成**（自动注册）：`import flag_gems` 时自动覆盖 `aten::argmax`，无需额外调用。
修改文件：`src/flag_gems/runtime/backend/_arm/ops/__init__.py`

---

## 四、潜在优化方向

| 优先级 | 算子/场景 | 当前状态 | 建议 |
|--------|----------|---------|------|
| **高** | `silu` (BF16) | BF16→FP32→Triton→BF16，2 次额外 cast | 实现 native BF16 silu kernel（Triton 支持 BF16 dot），消除 intermediate tensor |
| **高** | `mul` (SwiGLU gate) | ATen（主动排除） | 考虑 fused `silu_and_mul` kernel：一次 kernel 完成 silu(gate) * up，减少 memory traffic |
| **中** | `scaled_dot_product_attention` | ATen（-16% 已排除） | 调查回退原因（可能是小 batch/head_dim=128 形状不匹配）；尝试针对 decode M=1 的专用 attention kernel |
| **中** | `add`（residual） | ATen（主动排除） | 测试是否可 fuse 进 LayerNorm（pre-norm + add fused kernel） |
| **低** | `mean` / `rsqrt` (decode) | numpy（numel<4096，合理） | numpy 路径（~2µs）已快于 Triton 启动开销（~17µs），threshold 设置合理，无需修改 |
| **低** | `addmm` | 已注册，Qwen3 未调用 | 对有 bias 的其他模型有用，保留 |
| **低** | `softmax` (CPU) | numpy fallback | 仅在非 SDPA 路径生效，当前 Qwen 系列不触达 |

---

## 六、环境说明

| 组件 | 版本/路径 |
|------|---------|
| Python | 3.11 |
| PyTorch (venv-flagtree) | 2.10.0+cpu |
| PyTorch (venvllm) | 2.10.0+cpu |
| Transformers (venv-flagtree) | 5.2.0 |
| Transformers (venvllm) | 4.57.6 |
| Triton-CPU | commit 96fc8730（arm64-dev），SVE2 i8mm，OMP fix |
| FlagTree | v3.3.0x，min_dot_size=(1,4,4) |
| FlagGems | Phase1-4（M分支 + 融合 dequant + weight tiling + M-padding） |
| llama.cpp | b5929（/usr/share/cix/bin/llama-bench）|
| INT8 模型格式 | PyTorch `quantize_dynamic`（per-tensor）；llama.cpp Q8_0（per-block=32）|
| BF16 模型源 | ModelScope `/home/kevin/.cache/modelscope/hub/models/Qwen/` |

---

## 七、附录：Qwen 模型维度

| 模型 | hidden | intermediate | heads | head_dim |
|------|--------|-------------|-------|---------|
| Qwen3-0.6B | 1024 | 3072 | 16 | 128 |
| Qwen3-1.7B | 2048 | 6144 | 16 | 128 |
| Qwen2.5-1.5B | 1536 | 8960 | 12 | 128 |
| Qwen2-7B | 3584 | 18944 | 28 | 128 |

> 注：`mean`/`rsqrt` 的 numel=4096 阈值对所有模型 decode（M=1，形状 [1,hidden]）均触发 numpy 路径：最大 hidden=3584 < 4096。prefill（M>1）时形状更大，走 Triton。
