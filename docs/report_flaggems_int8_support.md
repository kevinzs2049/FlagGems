# 报告2：FlagGems Int8 支持现状分析

## 1. 范围说明

这里的 “Int8 支持” 分三类：

1. **普通张量 dtype=int8**（`torch.int8`）参与算子计算  
2. **量化张量**（`x.is_quantized=True`，如 `qint8/quint8`）  
3. **量化推理专用路径**（例如 vLLM 的 `cutlass_scaled_mm`）

---

## 2. 总体结论（先给结论）

1. FlagGems 并非“全局 Int8 框架”；当前是**部分算子可用**、**部分算子回退**、**部分明确不支持**。  
2. 在 NVIDIA 侧，真正有明确 Int8 高性能定位的是 `cutlass_scaled_mm` 路径（且目前主要是 SM90）。  
3. `is_quantized` 张量支持并不完整：有的算子回退到 PyTorch，有的直接 `NotImplementedError`。  

---

## 3. 明确的 Int8/量化相关能力

## 3.1 `cutlass_scaled_mm`（vLLM 常见量化线性路径）

文件：`src/flag_gems/fused/cutlass_scaled_mm.py`

- `dispatch_scaled_mm(...)` 明确支持 `a.dtype == torch.int8` 分支
- 但架构分发现状是：
  - `SM90`：有实现（`cutlass_scaled_mm_sm90_int8`）
  - `SM80/SM89/SM100/SM120`：当前 `NotImplemented`

这条路径在 vLLM patch 中被接到：

- `_C.cutlass_scaled_mm -> custom_cutlass_scaled_mm`  
  文件：`src/flag_gems/patches/patch_vllm_all.py`

> 结论：这是当前最实用的 Int8 高性能路径，但架构覆盖不完整。

## 3.2 Int8 作为普通输入 dtype（非量化张量）

部分算子可接受 `torch.int8`（或内部使用 int8 索引/键）：

- `topk/randperm/index/full` 等文件中存在 `torch.int8` 分支或常量
- 这类更多是“算子兼容性”，不等同于量化推理优化

---

## 4. 明确不支持或有限支持的点

## 4.1 量化张量 (`is_quantized`) 支持不完整

- `to_copy`：遇到 `x.is_quantized` 直接报错  
  文件：`src/flag_gems/ops/to.py`
- `isclose`：量化张量直接报错  
  文件：`src/flag_gems/ops/isclose.py`
- `copy_`：量化张量不走 FlagGems Triton，回退 `aten.copy_`  
  文件：`src/flag_gems/ops/copy.py`

> 结论：`is_quantized=True` 并非通用支持，很多场景依赖回退或不支持。

## 4.2 通用 GEMM (`mm`) 不是 Int8 目标实现

文件：`src/flag_gems/ops/mm.py`

- 类型优先级只列了 `float16/bfloat16/float32`
- `mm` 主路径按浮点 dtype 设计

> 结论：Int8 GEMM 不是通用 `mm` 路径主目标，要走 `cutlass_scaled_mm` 这类专用接口。

---

## 5. FP8 与 Int8 的关系（避免混淆）

当前代码里有较多 `fp8` / `uint8` 相关逻辑（例如 KV cache、`per_token_group_quant_fp8`）：

- `src/flag_gems/ops/per_token_group_quant_fp8.py`
- `src/flag_gems/fused/concat_and_cache_mla.py`

这类是 **FP8 量化链路**，不是 Int8 算子本身。  
`uint8` 常用于 bitcast/storage，不等于 Int8 算术优化。

---

## 6. 对实际使用的建议

1. 若目标是 NVIDIA 上 vLLM Int8 推理，优先评估并绑定 `cutlass_scaled_mm` 路径。  
2. 不要假设 `torch.int8`/`is_quantized` 在所有 FlagGems 算子都可直接用。  
3. 对关键链路做两步验证：  
   - 功能：是否命中 FlagGems 路径（dispatch + 调用计数）  
   - 性能：是否优于 aten/native（同形状同 batch 的 profiler 对比）  

---

## 7. 一句话总结

FlagGems 目前对 Int8 是“**重点场景专用支持 + 通用场景部分兼容**”，不是“全算子全链路 Int8 完整支持”。对于 vLLM，最有价值的是 `cutlass_scaled_mm` 这条专用量化路径。

---

## 8. ARM64 上 Int4 迁移潜力（新增）

基于当前代码，Int4 在 FlagGems 主仓还没有独立实现入口，但 vLLM CPU 侧已经有可参考的 ARM64 路径：

- `/home/kevin/vllm_source/csrc/moe/dynamic_4bit_int_moe_cpu.cpp`
  - 在 `__aarch64__` 下调用 `at::_ops::_dyn_quant_matmul_4bit`
- `/home/kevin/vllm_source/csrc/cpu/torch_bindings.cpp`
  - 注册 `dynamic_4bit_int_moe` CPU op

这意味着 ARM64 Int4 不是“从零开始”，而是可以先对齐现有 4-bit matmul 合同（packing/scale/group_size），再把 FlagGems `_arm` 内核逐步接入模型热点路径。

可执行迁移清单见：

- `docs/arm64_int8_int4_migration_checklist.md`
