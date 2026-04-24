"""
attention.py — ARM CPU Flash Attention (Triton-CPU)

Flash Attention v2 在线 softmax，无需 O(M×N) 中间矩阵。
支持：GQA (grouped-query attention), is_causal, BF16 输入。

性能 (M=512, D=128, H=16, OMP=6, CIX P1 CD8180):
  ATen:  ~179ms  →  Triton:  ~40ms  (4.5x speedup)
  BLOCK_M=32, BLOCK_N=16 最优（经 sweep 验证）

Decode (M < BLOCK_M=32) 自动回落 ATen：tl.dot 要求 M≥4。
非 BF16 或带 attn_mask 时同样回落 ATen。
"""
import ctypes
import logging
import os

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

log = logging.getLogger(__name__)

# ── libsleef 预加载 (tl.math.exp2 在 Triton-CPU .so 里依赖 SLEEF) ─────────

def _ensure_sleef():
    try:
        import triton as _t
        sleef_dir = os.path.join(os.path.dirname(_t.__file__), "_C")
        sleef_so  = os.path.join(sleef_dir, "libsleef.so.3")
        if not os.path.exists(sleef_so):
            return
        ld = os.environ.get("LD_LIBRARY_PATH", "")
        if sleef_dir not in ld:
            os.environ["LD_LIBRARY_PATH"] = f"{sleef_dir}:{ld}"
        ctypes.CDLL(sleef_so)           # 预加载到进程，后续 dlopen 可找到符号
    except Exception:
        pass

_ensure_sleef()

# 保存原始 ATen SDPA，内部 fallback 时使用（避免 monkey-patch 后无限递归）
_aten_sdpa = F.scaled_dot_product_attention

# log2(e) = 1/ln(2) — 用于 exp2 代替 exp（避免 SLEEF 精度损失）
_LOG2E: float = 1.44269504089

# ── 块大小（经 sweep 验证：BLOCK_N=16 最优）─────────────────────────────────
_BLOCK_M: int = 32
_BLOCK_N: int = 16

# ── Flash Attention Triton Kernel ───────────────────────────────────────────

@triton.jit
def _flash_attn_fwd_kernel(
    Q, K, V, sm_scale, Out,
    # [B*Hq, M, D]
    stride_qh, stride_qm, stride_qk,
    # [B*Hkv, N, D]
    stride_kh, stride_kn, stride_kk,
    # [B*Hkv, N, D]
    stride_vh, stride_vn, stride_vk,
    # [B*Hq, M, D]
    stride_oh, stride_om, stride_ok,
    seqlen_q, seqlen_k,
    q_numhead, kv_numhead,          # GQA 支持
    LOG2E: tl.constexpr,            # 1.44269504
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    IS_CAUSAL: tl.constexpr,        # 编译时常量，生成两条代码路径
):
    pid_bh = tl.program_id(0)       # batch × Q-head (合并)
    pid_m  = tl.program_id(1)       # M-tile 索引

    # GQA 映射：每 (Hq//Hkv) 个 Q-head 共享一个 KV-head
    head_id    = pid_bh % q_numhead
    batch_id   = pid_bh // q_numhead
    kv_head_id = head_id * kv_numhead // q_numhead   # 正确的 GQA 映射

    Q_bh = Q   + (batch_id * q_numhead  + head_id)    * stride_qh
    K_bh = K   + (batch_id * kv_numhead + kv_head_id) * stride_kh
    V_bh = V   + (batch_id * kv_numhead + kv_head_id) * stride_vh
    O_bh = Out + (batch_id * q_numhead  + head_id)    * stride_oh

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, HEAD_DIM)
    mask_m = offs_m < seqlen_q

    # Q: [BLOCK_M, HEAD_DIM]，预乘 sm_scale*LOG2E（转入 log2 域）
    q = tl.load(
        Q_bh + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk,
        mask=mask_m[:, None], other=0.0,
    ).to(tl.float32) * (sm_scale * LOG2E)

    # 在线 softmax 状态（per-row, log2 域）
    m_i = tl.full([BLOCK_M], float('-inf'), dtype=tl.float32)
    lse = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    # Causal：只遍历到当前 Q-tile 位置
    if IS_CAUSAL:
        kv_end = tl.minimum(seqlen_k, (pid_m + 1) * BLOCK_M)
    else:
        kv_end = seqlen_k

    for start_n in range(0, kv_end, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < seqlen_k

        # K^T: [HEAD_DIM, BLOCK_N]（交换 k/n offset 实现转置加载）
        k = tl.load(
            K_bh + offs_k[:, None] * stride_kk + offs_n[None, :] * stride_kn,
            mask=mask_n[None, :], other=0.0,
        ).to(tl.float32)

        # QK^T: [BLOCK_M, HEAD_DIM] × [HEAD_DIM, BLOCK_N] → [BLOCK_M, BLOCK_N]
        # q 已在 log2 域（含 sm_scale*LOG2E），结果直接可用 exp2
        qk = tl.dot(q.to(tl.bfloat16), k.to(tl.bfloat16)).to(tl.float32)

        if IS_CAUSAL:
            causal_ok = offs_m[:, None] >= offs_n[None, :]
            qk = tl.where(causal_ok & mask_n[None, :], qk, float('-inf'))
        else:
            qk = tl.where(mask_n[None, :], qk, float('-inf'))

        # 在线 softmax（log2 域）
        m_new = tl.maximum(m_i, tl.max(qk, axis=1))       # [BLOCK_M]
        alpha  = tl.math.exp2(m_i - m_new)                  # 旧行缩放
        p      = tl.math.exp2(qk - m_new[:, None])         # [BLOCK_M, BLOCK_N]

        lse = lse * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]

        # V: [BLOCK_N, HEAD_DIM]
        v = tl.load(
            V_bh + offs_n[:, None] * stride_vn + offs_k[None, :] * stride_vk,
            mask=mask_n[:, None], other=0.0,
        ).to(tl.bfloat16)

        # P @ V: [BLOCK_M, BLOCK_N] × [BLOCK_N, HEAD_DIM] → [BLOCK_M, HEAD_DIM]
        acc = tl.dot(p.to(tl.bfloat16), v, acc=acc)
        m_i = m_new

    # 归一化 + 写回
    acc = acc / lse[:, None]
    tl.store(
        O_bh + offs_m[:, None] * stride_om + offs_k[None, :] * stride_ok,
        acc.to(tl.bfloat16),
        mask=mask_m[:, None],
    )


# ── Python 包装器 ────────────────────────────────────────────────────────────

def _triton_flash_attn(
    query: torch.Tensor,
    key:   torch.Tensor,
    value: torch.Tensor,
    sm_scale: float,
    is_causal: bool,
) -> torch.Tensor:
    """核心 Triton kernel 调用，调用前已确认可以走 Triton 路径。"""
    B, Hq, M, D = query.shape
    Hkv = key.shape[1]

    # 合并 batch+head → [B*H, seq, D]
    q   = query.reshape(B * Hq,  M, D)
    k   = key.reshape(B * Hkv, -1, D)
    v   = value.reshape(B * Hkv, -1, D)
    N   = k.shape[1]
    out = torch.empty_like(q)

    grid = (B * Hq, triton.cdiv(M, _BLOCK_M))

    _flash_attn_fwd_kernel[grid](
        q, k, v, sm_scale, out,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        M, N,
        Hq, Hkv,
        _LOG2E,
        BLOCK_M=_BLOCK_M, BLOCK_N=_BLOCK_N, HEAD_DIM=D,
        IS_CAUSAL=is_causal,
    )
    return out.reshape(B, Hq, M, D)


def scaled_dot_product_attention(
    query,
    key,
    value,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    scale=None,
    enable_gqa=False,
):
    """
    aten::scaled_dot_product_attention — ARM CPU Flash Attention 版本。

    Triton 路径条件（否则回落 ATen）：
      - dtype = bfloat16
      - attn_mask = None
      - dropout_p = 0.0
      - seqlen_q >= BLOCK_M (=32)
      - head_dim in {16,32,64,128,256}
    """
    B, Hq, M, D = query.shape

    # M=1 decode fast path: C runtime flash_attn_decode_bf16 via triton-cpu.
    # Measured +1.2% E2E on Qwen3-1.7B INT8 vs ATen fallback (3 rounds A/B).
    # Requires BF16, no mask, no dropout, contiguous Q/K/V.
    if (M == 1 and B == 1
            and query.dtype == torch.bfloat16
            and attn_mask is None
            and dropout_p == 0.0
            and query.is_contiguous()
            and key.is_contiguous() and value.is_contiguous()):
        try:
            from triton.language.extra.cpu.runtime import flash_attn_decode_bf16
        except ImportError:
            flash_attn_decode_bf16 = None
        if flash_attn_decode_bf16 is not None:
            Hkv = key.shape[1]
            seq_len = key.shape[2]
            sm_scale = scale if scale is not None else D ** -0.5
            q_flat = query.squeeze(0).squeeze(1).contiguous()
            k_flat = key.squeeze(0).contiguous()
            v_flat = value.squeeze(0).contiguous()
            out_flat = torch.empty(Hq, D, dtype=torch.bfloat16)
            flash_attn_decode_bf16(
                q_flat, k_flat, v_flat, out_flat,
                seq_len, D, sm_scale, Hq, Hkv,
                k_flat.stride(1), v_flat.stride(1),
            )
            return out_flat.unsqueeze(0).unsqueeze(2)

    # Prefill fast path: Triton Flash Attention kernel (requires M >= BLOCK_M).
    use_triton = (
        query.dtype == torch.bfloat16
        and attn_mask is None
        and dropout_p == 0.0
        and M >= _BLOCK_M
        and D in {16, 32, 64, 128, 256}
    )

    if not use_triton:
        log.debug("GEMS SDPA: ATen fallback (M=%d, dtype=%s, mask=%s)",
                  M, query.dtype, attn_mask is not None)
        return _aten_sdpa(
            query, key, value,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale,
            enable_gqa=enable_gqa,
        )

    sm_scale = scale if scale is not None else D ** -0.5
    log.debug("GEMS SDPA: Triton Flash Attention (M=%d, N=%d, D=%d, causal=%s, Hq=%d, Hkv=%d)",
              M, key.shape[2], D, is_causal, Hq, key.shape[1])
    return _triton_flash_attn(query, key, value, sm_scale, is_causal)
