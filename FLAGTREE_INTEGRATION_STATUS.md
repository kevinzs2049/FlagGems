# FlagGems + FlagTree CPU 集成工作总结

## 概述

本文档记录了FlagGems与FlagTree CPU后端集成的进展、遇到的问题及解决方案。

## 环境信息

| 项目 | 路径 |
|------|------|
| FlagGems | `/home/kevin/FlagGems` |
| FlagTree | `/home/kevin/FlagTree` |
| triton-cpu | `/home/kevin/triton-cpu` |

**环境变量**:
- `FLAGTREE_BACKEND=cpu`
- `GEMS_VENDOR=arm`
- Python 3.13, PyTorch CPU版本

**测试命令**:
```bash
cd /home/kevin/FlagGems
python3 -S test_qwen3_flagtree.py
```

---

## 已完成的工作

### 1. 修复 `torch.cpu.default_generators` 问题

**问题**: CPU没有CUDA的`default_generators`属性，导致随机数相关ops失败。
```
AttributeError: module 'torch.cpu' has no attribute 'default_generators'
```

**解决方案**: 将随机数相关ops添加到`CUSTOMIZED_UNUSED_OPS`列表，使用PyTorch原生实现。

**修改文件**: `/home/kevin/FlagGems/src/flag_gems/runtime/backend/_arm/__init__.py`

**添加的ops**:
```python
# Random number ops - CPU has no torch.cpu.default_generators
"dropout",
"exponential_",
"multinomial",
"normal",
"normal_",
"normal_tensor_tensor",
"normal_tensor_float",
"normal_float_tensor",
"rand",
"rand_like",
"randn",
"randn_like",
"randperm",
"uniform_",
```

### 2. 修复 RMSNorm Monkey Patch 问题

**问题**: `patch_qwen3_rmsnorm()`的类检查逻辑错误，检查实例属性`weight`而非类属性。

**解决方案**: 修改检查逻辑为检查`__init__`签名中的`variance_epsilon`参数。

**修改文件**: `/home/kevin/FlagGems/src/flag_gems/runtime/backend/_arm/fused/patch_qwen3_rmsnorm.py`

**修改内容**:
```python
def patch_class(cls) -> bool:
    """Replace cls.forward with the fused kernel. Returns True if patched."""
    if cls in _PATCHED_CLASSES:
        return False
    # Check if this looks like an RMSNorm class (has variance_epsilon in __init__)
    init_sig = getattr(cls.__init__, "__signature__", None)
    if init_sig:
        params = list(init_sig.parameters.keys())
        if "variance_epsilon" not in params:
            logger.debug("patch_class: %s has no variance_epsilon param, skipping", cls.__name__)
            return False
    cls._original_forward = cls.forward
    cls.forward = _make_triton_forward(cls.__name__)
    _PATCHED_CLASSES.add(cls)
    logger.info("Patched %s.forward with fused Triton RMSNorm", cls.__name__)
    return True
```

### 3. 基础Ops测试通过

以下ops已验证正确：
- **算术运算**: add, mul, sub, neg, rsqrt
- **矩阵运算**: mm, bmm, addmm, matmul
- **归一化**: softmax, layer_norm, RMSNorm
- **其他**: mean, embedding, index_select, where, contiguous

**RMSNorm Kernel测试结果**:
```
RMSNorm direct test: max_diff=0.00000000 PASS
RMSNorm bfloat16 test: max_diff=0.00781250 PASS
```

---

## 当前存在的问题

### 核心问题：模型推理结果非确定性

**现象**:
```
Forward pass 1: logits sum=-177152.000000
Forward pass 2: logits sum=-229376.000000
Forward pass 3: logits sum=-174080.000000
All sums equal: False
```

**Qwen3-0.6B测试结果**:
```
[test] REF: 3.41s, 8 tokens, 2.349 tok/s
[test] GEMS: 3.30s, 8 tokens, 2.426 tok/s
[test] ACCURACY: FAIL maxdiff=4270.0
[test] REF: 'What is your name? What is your role? What is your'
[test] GEMS: 'What is your name? nothingness1genETatoeat'
```

**Layer-by-layer分析**:
```
Layer 0: max_diff=0.00000000 PASS
Layer 1: max_diff=0.02343750 FAIL  <- 问题从Layer 1开始出现
```

**关键发现**:
- 单个op测试（mm, bmm, addmm, mean, RMSNorm等）都是确定性的
- 完整模型forward pass结果每次不同
- RMSNorm patch正确工作，Layer 0（embedding层）输出正确

---

## 已尝试的修复

### 1. 禁用Attention Ops

**原因**: ARM backend的attention使用`@triton.autotune`，可能产生非确定性行为。

**修改**: 添加到`CUSTOMIZED_UNUSED_OPS`:
```python
# Attention ops with autotune cause non-deterministic behavior on CPU
"scaled_dot_product_attention",
"scaled_dot_product_attention_forward",
"scaled_dot_product_attention_backward",
```

**结果**: 问题仍然存在

### 2. 禁用随机数Ops

**原因**: 随机数ops可能引入不确定性。

**结果**: 问题仍然存在

---

## 怀疑的原因

### 1. `@triton.autotune` 在CPU上的问题

ARM backend有13个ops使用`@triton.autotune`:
```
mean, sum, silu, addmm, max, all, any, attention, bmm, isin, log_softmax, min, softmax
```

autotune需要benchmark选择最优配置，在CPU上可能产生非确定性行为。

### 2. FlagTree编译器的潜在Bug

某些kernel编译后可能有：
- 未初始化内存
- 循环展开问题
- LLVM优化问题

### 3. Ops组合问题

单独测试每个op都正确，但组合使用时可能出现问题（如内存复用、状态累积等）。

---

## triton-cpu tl.dot精度验证

之前怀疑triton-cpu的`tl.dot`有精度问题，经过详细测试验证：

**结论**: **不是编译器Bug，是用户代码错误**

**错误代码**:
```python
b_ptrs += BLOCK_K  # 错误：忽略了stride_bk
```

**正确代码**:
```python
b_ptrs += BLOCK_K * stride_bk  # 对于行主序B[K,N]，stride_bk=N
```

详见：`/home/kevin/triton-cpu-tol-dot.bugreport.md`

---

## 相关文件清单

| 文件 | 说明 |
|------|------|
| `/home/kevin/FlagGems/src/flag_gems/runtime/backend/_arm/__init__.py` | ARM backend配置，`CUSTOMIZED_UNUSED_OPS`列表 |
| `/home/kevin/FlagGems/src/flag_gems/runtime/backend/_arm/fused/fused_add_rms_norm.py` | RMSNorm融合kernel实现 |
| `/home/kevin/FlagGems/src/flag_gems/runtime/backend/_arm/fused/patch_qwen3_rmsnorm.py` | Qwen3 RMSNorm monkey patch |
| `/home/kevin/FlagGems/src/flag_gems/runtime/backend/_arm/ops/*.py` | ARM优化的ops实现 |
| `/home/kevin/FlagGems/src/flag_gems/runtime/backend/_arm/tune_configs.yaml` | autotune配置 |
| `/home/kevin/FlagGems/test_qwen3_flagtree.py` | Qwen3-0.6B测试脚本 |
| `/home/kevin/triton-cpu-tol-dot.bugreport.md` | tl.dot精度问题分析报告 |

---

## 待调查项

1. **检查所有`@triton.autotune` ops** - 可能需要禁用或替换为固定配置
2. **检查FlagTree编译器** - 查看是否有未初始化内存或循环问题
3. **检查ops之间的状态累积** - 是否有全局状态被意外修改
4. **检查内存管理** - 是否有内存复用导致的问题

---

## 测试用例

### 基础Op测试
```python
import torch
import flag_gems
flag_gems.enable()

# Add
x = torch.randn(5000, device='cpu', dtype=torch.float32)
y = torch.randn(5000, device='cpu', dtype=torch.float32)
z = x + y  # max_diff=0.00000000 PASS

# MatMul
a = torch.randn(64, 64, device='cpu', dtype=torch.float32)
b = torch.randn(64, 64, device='cpu', dtype=torch.float32)
c = torch.mm(a, b)  # max_diff=0.00000000 PASS

# RMSNorm
from flag_gems.runtime.backend._arm.fused import rms_norm_forward
x = torch.randn(1, 5, 1024, device='cpu', dtype=torch.float32)
weight = torch.ones(1024, device='cpu', dtype=torch.float32)
y = rms_norm_forward(x, [1024], weight, 1e-6)  # PASS
```

### 非确定性测试
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")
model.to('cpu').eval()

from flag_gems.runtime.backend._arm.fused import patch_qwen3_rmsnorm
patch_qwen3_rmsnorm()
import flag_gems
flag_gems.enable()

inputs = tokenizer('Hello', return_tensors='pt')
for i in range(3):
    with torch.no_grad():
        out = model(**inputs)
        print(f'Pass {i+1}: logits sum={out.logits.sum().item():.6f}')
# 每次结果不同 -> 非确定性问题
```

---

## 更新历史

| 日期 | 内容 |
|------|------|
| 2026-03-04 | 初始文档，记录集成进展和当前问题 |
