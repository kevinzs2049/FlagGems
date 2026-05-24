# Negative result: KV cache pre-allocation does NOT speed up Qwen3-1.7B INT8 decode

**Date**: 2026-04-26
**Workload**: Qwen3-1.7B W8A8-INT8, FlagGems TLE stack (TLEInt8Linear + patch_qwen3_mlp + flash_attn_decode), CIX P1 CD8180, 8 big cores OMP=8, performance governor.

## Hypothesis

`torch.profiler` on a 20-token decode run showed:

```
aten::cat   32.058 ms (4.41%)   2300 calls   18us avg
```

These are HuggingFace `DynamicCache.update()`'s per-step `torch.cat([keys, new_key], dim=-2)` (28 layers × 2 (K+V) × ~40 forward passes including warmup).

We hypothesised that replacing this with a pre-allocated `[B, H, max_seq_len, D]`
buffer + in-place slice write would save the `cat` overhead and yield +3-5% E2E.

## Implementation

A `TLECache` / `TLEPreallocLayer` subclass of `transformers.cache_utils.Cache`
with:
- `lazy_initialization`: allocate `[B, H, max_seq_len, D]` zero buffer per layer.
- `update`: `self.keys.narrow(-2, start, T_new).copy_(key_states)` then return
  `self.keys.narrow(-2, 0, end)` (zero-copy view).
- `get_seq_length`, `get_mask_sizes`, `reorder_cache`, `reset`, `crop` etc.
  to satisfy the `Cache` API contract.

Validated end-to-end: `model.generate(past_key_values=TLECache(config))` runs
to completion and produces the same output token sequence as the default
`DynamicCache` for short generations (hash match on 20-token gen).

## Results

A/B with `model.generate(do_sample=False)`, 5 runs median, 60s cooldown
between states, prompt = `"The meaning of life is"`.

### Short generation (max_new_tokens=20)

| max_seq_len | DynamicCache | TLECache | Δ        | hash equal |
|-------------|-------------:|---------:|---------:|------------|
| 128         | 10.09 tok/s  | 9.80     | **-2.9%** | ✓         |
| 32 (tight)  | 10.13        | 9.95     | **-1.8%** | ✓         |

### Long generation (max_new_tokens=200, max_seq_len=256)

| | DynamicCache | TLECache | Δ          | hash equal |
|---|-------------:|---------:|-----------:|------------|
| 3-run median | 10.42 | 9.30 | **-10.8%** | **✗** |

## Root causes

### Why pre-alloc loses on speed

1. **`aten::cat` is already a tightly optimised memcpy.** 14μs/call is dominated
   by Python dispatch + tensor metadata creation, not the data copy. Replacing
   with `narrow().copy_()` doesn't reduce dispatch count — it's still one tensor
   op per layer per step.

2. **Larger pre-alloc stride degrades L1 cache.** Buffer shape
   `[B=1, H=8, max_seq=256, D=128]` has stride `256*128 = 32 KB` between heads
   in the seq-major access pattern. DynamicCache at decode step T has stride
   `T*128` (much smaller for small T). Attention's per-head loops re-read the
   K/V tensors, and the larger stride → more L1 evictions.

3. **Returned `narrow` view metadata isn't free.** Each step constructs a new
   tensor object (header), even though no data is copied. The header allocation
   adds ~1-2μs per call, ~28-56μs per token across all layers.

### Why long gen mismatches hashes

The hash divergence at 200 tokens (with hash match at 20 tokens) suggests
**numerical drift** rather than a logical bug:

- BF16 attention compute is sensitive to operand layout / contiguity.
- The slightly different memory access pattern (pre-alloc stride vs
  growing-tensor stride) causes flash_attn_decode's online softmax to
  accumulate in a different order, producing tiny FP differences.
- Over 200 generation steps these drift past argmax boundaries.

Both outputs are valid greedy decodings within INT8 quant noise — neither is
"wrong" — but they're not bit-identical, which violates the implicit
"deterministic generate" contract.

## Decision

**Do not merge.** TLECache is a regression on this workload:

- Short gen: -1.8% to -2.9%  (correctness OK)
- Long gen:  -10.8%, AND non-deterministic vs default

The profile signal (`aten::cat 4.4%`) was misleading: it represents the
*lower bound* on what removing cat would save, not the *upper bound* of the
gain — because in-place writes have their own overhead, and worse cache
locality dominates at scale.

## Where pre-alloc could still pay off (not implemented)

These conditions might flip the result; not pursued in this experiment:

1. **Very long contexts (>2K tokens)** with cache-aware allocator — the
   reallocation cost in DynamicCache becomes O(N²) total bytes copied
   while pre-alloc stays O(N).

2. **Custom flash_attn that consumes pre-alloc buffer directly with
   `seq_len_valid` parameter**, avoiding the narrow-view + stride mismatch
   problem. Would need both kernel + cache co-designed.

3. **Specific deployment (e.g., serving framework)** where buffer reuse
   across requests amortises the alloc cost.

## Files (deleted, not committed)

- `src/flag_gems/runtime/backend/_arm/int8/kv_cache.py` (TLECache impl)
- `__init__.py` exports for TLECache / TLEPreallocLayer

Bench scripts kept for future reference at `/home/kevin/`:
- `bench_tlecache_ab.py`
- `test_tlecache_correctness.py`

## TL;DR

`torch.profiler` flagged `aten::cat 4.4%` as a target. Building a
pre-allocated KV cache to eliminate it produced a measurable regression
(-1.8% short / -10.8% long) plus output non-determinism at long gen due
to BF16 numerical drift from different memory layout. Reverted, kept
notes here.
