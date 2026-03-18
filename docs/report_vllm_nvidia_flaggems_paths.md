# 报告1：NVIDIA 架构下 vLLM + FlagGems 算子路径分析

## 1. 范围与结论

本报告基于代码静态分析（`FlagGems` + `vLLM` 源码）总结 **NVIDIA GPU** 下 vLLM 与 FlagGems 的算子调用路径。核心结论：

1. vLLM 在 NVIDIA 上先选择 attention backend，再决定是否命中 FlagGems patch。  
2. FlagGems 对 vLLM 的接入分两条线：  
   - `flag_gems.enable(...)`：接管 ATen 算子（`aten::*`）  
   - `flag_gems.apply_gems_patches_to_vllm(...)`：接管 vLLM 自定义模块/自定义算子（`_C::*`, `_moe_C::*` 等）
3. 注意力主路径并不总是走 FlagGems；是否命中取决于 vLLM backend（如 `FLASH_ATTN`/`TRITON_ATTN`/`FLASHINFER`）。

补充：本报告对应的机器可读映射表已输出到：

- `docs/nvidia_vllm_flaggems_operator_paths.csv`

---

## 2. backend 选择：先看 vLLM 选谁

NVIDIA 平台 backend 选择逻辑在：

- `/home/kevin/vllm_source/vllm/platforms/cuda.py`
- `/home/kevin/vllm_source/vllm/v1/attention/selector.py`

默认优先级（非 MLA）大致是：

- SM90（Hopper）: `FLASHINFER` > `FLASH_ATTN` > `TRITON_ATTN` > `FLEX_ATTENTION`
- 非 SM90: `FLASH_ATTN` > `FLASHINFER` > `TRITON_ATTN` > `FLEX_ATTENTION`

可通过 `VLLM_ATTENTION_BACKEND` 强制指定。

---

## 3. vLLM attention 路径与 FlagGems 对接关系

`vllm::unified_attention_with_output` 只是壳，实际执行 `self.impl.forward(...)`，见：

- `/home/kevin/vllm_source/vllm/attention/layer.py`

因此真正路径取决于 `impl` 类型：

| vLLM backend/impl | 默认实现 | FlagGems patch 覆盖情况 | 结果路径 |
|---|---|---|---|
| `FlashAttentionImpl` | vLLM flash-attn 路径 | **有**（patch `forward`） | 走 `custom_gems_flash_attention_impl_forward` |
| `TritonAttentionImpl` | vLLM Triton attention | 当前无直接替换 | 仍走 vLLM 原生 Triton |
| `FlashInferBackend` | vLLM FlashInfer | 当前无直接替换 | 仍走 vLLM 原生 FlashInfer |
| `CPUAttentionBackendImpl` | vLLM CPU `_C::cpu_attention_with_kv_cache` | 当前无默认替换 | 仍走 vLLM CPU C++ |

FlagGems 对 `FlashAttentionImpl.forward` 的 patch 在：

- `src/flag_gems/patches/patch_vllm_all.py`

patched forward 内部主要调用：

- `flag_gems.reshape_and_cache_flash(...)`
- `flag_gems.flash_attn_varlen_func(...)`

其中 `flash_attn_varlen_func` 再分支：

- `use_c_extension=True`：走 `torch.ops.flag_gems.flash_attn_varlen_func`（C++ 扩展）
- `use_c_extension=False`：走 FlagGems Triton/Python 实现  

见：`src/flag_gems/ops/attention.py`

---

## 4. vLLM 其他算子路径（非 attention）

`apply_gems_patches_to_vllm` 做两类 patch：

### 4.1 patch vLLM Python 类方法

在 `src/flag_gems/patches/patch_vllm_all.py` 中，按可导入性动态 patch：

- `RMSNorm.forward_cuda`
- `RotaryEmbedding.forward_cuda`
- `PagedAttention.write_to_paged_cache`
- `SiluAndMul.forward_cuda`
- `TritonMLAImpl._forward_decode`
- `FlashAttentionImpl.forward`
- `FlashAttnMLAImpl._forward_decode`

这些方法被替换后，内部调用 FlagGems 模块/融合算子。

### 4.2 patch vLLM 自定义库算子 (`torch.ops._C/*`)

同文件通过 `patch_vllm_lib` patch：

- `_C.silu_and_mul`
- `_C.cutlass_scaled_mm`
- `_C.per_token_group_fp8_quant`
- `_C.apply_repetition_penalties_`
- `_moe_C.moe_align_block_size`
- `_moe_C.topk_softmax`
- `_moe_C.grouped_topk`
- `_moe_C.moe_sum`
- `_vllm_fa3_C.get_scheduler_metadata`
- `_C_cache_ops.concat_and_cache_mla`

这些不是 `aten::*`，而是 vLLM 自己的扩展算子入口。

---

## 5. `aten::*` 路径：FlagGems 如何接管

调用 `flag_gems.enable(...)` 后，`src/flag_gems/__init__.py` 的 `_FULL_CONFIG` 会把大量 `aten::*` 映射到 FlagGems 实现（通常是 Triton/Python）。

但如果启用了 C extension 且存在 `aten_patch`，会有一小批 `aten::*` 被 C++ 直接接管（并从 Python 注册列表排除）：

- `max.dim_max`
- `max.dim`
- `max`
- `sum`
- `zeros`
- `fill.Scalar`
- `fill_.Scalar`

相关文件：

- `src/flag_gems/csrc/aten_patch.cpp`
- `src/flag_gems/config.py`
- `src/flag_gems/runtime/register.py`

---

## 6. 三条最终执行路径（迁移 ARM64 时可直接复用的框架）

### 路径A：ATen -> FlagGems Triton/Python

- 触发：`flag_gems.enable(...)`
- 典型：`aten::add/mul/pow/mm/...`（不在 `aten_patch` 小集合时）

### 路径B：ATen -> FlagGems CUDA C++ (`aten_patch`)

- 触发：`USE_C_EXTENSION=1` 且扩展可用
- 仅少量 `aten::*`（上面 7 个）

### 路径C：vLLM 自定义 op/模块 -> FlagGems patch

- 触发：`flag_gems.apply_gems_patches_to_vllm(...)`
- 覆盖 vLLM `_C/_moe_C/...` 与部分 backend 实现类方法

---

## 7. 对 ARM64 移植的参考建议

1. 保持三层结构不变：  
   - ATen 层（`enable`）  
   - vLLM patch 层（`apply_gems_patches_to_vllm`）  
   - backend 选择层（vLLM 平台/attention backend）
2. 先做路径C（vLLM patch）再做路径A（ATen 覆盖），更容易定位收益来源。  
3. 对 attention，优先在目标 backend 的 `impl.forward` 做可回退 patch（带开关），避免一次性替全链路。  
4. 每次变更都保留两份证据：  
   - dispatch 证据（注册点）  
   - runtime 证据（调用计数/profiler）  
