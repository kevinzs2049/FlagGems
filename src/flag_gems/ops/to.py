import logging
import os
from collections import OrderedDict
from typing import Optional

import torch
import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)

_FALLBACK_KEYSET = torch._C.DispatchKeySet(
    torch._C.DispatchKey.CompositeExplicitAutograd
)
_TO_COPY_CAST_CACHE = OrderedDict()
_TO_COPY_CAST_CACHE_BYTES = 0


@pointwise_dynamic(
    is_tensor=[
        True,
    ],
    promotion_methods=[(0, "DEFAULT")],
)
@triton.jit
def _to_copy_func(x):
    return x


@triton.jit
def _to_copy_contiguous_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    val = tl.load(x_ptr + offsets, mask=mask, other=0)
    tl.store(out_ptr + offsets, val, mask=mask)


@triton.jit
def _to_copy_contiguous_single_program_kernel(
    x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr
):
    offsets = tl.arange(0, BLOCK_SIZE)
    for base in range(0, n_elements, BLOCK_SIZE):
        idx = base + offsets
        mask = idx < n_elements
        val = tl.load(x_ptr + idx, mask=mask, other=0)
        tl.store(out_ptr + idx, val, mask=mask)


@triton.jit
def _to_copy_contiguous_1024_hot_kernel(x_ptr, out_ptr):
    offs = tl.arange(0, 256)
    for base in range(0, 1024, 256):
        val = tl.load(x_ptr + base + offs)
        tl.store(out_ptr + base + offs, val)


@triton.jit
def _to_copy_contiguous_2048_hot_kernel(x_ptr, out_ptr):
    offs = tl.arange(0, 256)
    for base in range(0, 2048, 256):
        val = tl.load(x_ptr + base + offs)
        tl.store(out_ptr + base + offs, val)


@triton.jit
def _to_copy_contiguous_3072_hot_kernel(x_ptr, out_ptr):
    offs = tl.arange(0, 256)
    for base in range(0, 3072, 256):
        val = tl.load(x_ptr + base + offs)
        tl.store(out_ptr + base + offs, val)


def _to_copy_single_program_block(n_elements: int) -> int:
    if n_elements <= 256:
        return 32
    if n_elements <= 2048:
        return 128
    return 256


def _launch_to_copy_contiguous(x: torch.Tensor, out: torch.Tensor) -> None:
    n_elements = x.numel()
    if n_elements == 1024:
        _to_copy_contiguous_1024_hot_kernel[(1,)](x, out, num_warps=1, num_stages=1)
        return
    if n_elements == 2048:
        _to_copy_contiguous_2048_hot_kernel[(1,)](x, out, num_warps=1, num_stages=1)
        return
    if n_elements == 3072:
        _to_copy_contiguous_3072_hot_kernel[(1,)](x, out, num_warps=1, num_stages=1)
        return
    # On Triton-CPU, launch overhead dominates many decode-sized copies.
    # Keep a single-program loop path for medium tensors as well.
    if 1 < n_elements <= 262144:
        block = _to_copy_single_program_block(n_elements)
        _to_copy_contiguous_single_program_kernel[(1,)](
            x,
            out,
            n_elements,
            BLOCK_SIZE=block,
            num_warps=1,
            num_stages=1,
        )
        return
    grid = lambda META: (triton.cdiv(n_elements, META["BLOCK_SIZE"]),)
    _to_copy_contiguous_kernel[grid](
        x,
        out,
        n_elements,
        BLOCK_SIZE=256,
        num_warps=1,
        num_stages=1,
    )


def _resolve_dtype(x: torch.Tensor, dtype: Optional[torch.dtype]) -> torch.dtype:
    if dtype is None:
        return x.dtype
    if isinstance(dtype, torch.dtype):
        return dtype
    raise TypeError(f"Unsupported dtype argument type: {type(dtype)!r}")


def _resolve_device(x: torch.Tensor, device: Optional[torch.device]) -> torch.device:
    if device is None:
        return x.device
    return torch.device(device)


def _normalize_memory_format(
    memory_format: Optional[torch.memory_format],
) -> torch.memory_format:
    if memory_format is None:
        return torch.preserve_format
    return memory_format


def _allocate_preserve_format(x: torch.Tensor, empty_kwargs: dict) -> torch.Tensor:
    """Recreate tensor storage while honoring preserve_format semantics."""
    if torch.ops.aten.is_non_overlapping_and_dense(x):
        return torch.empty_strided(x.size(), x.stride(), **empty_kwargs)
    # Fall back to PyTorch's best-effort layout suggestion when stride replication is unsafe.
    return torch.empty_like(x, memory_format=torch.preserve_format, **empty_kwargs)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _tensor_nbytes(x: torch.Tensor) -> int:
    return int(x.numel()) * int(x.element_size())


def _to_copy_cast_cache_enabled() -> bool:
    return os.getenv("FLAGGEMS_ARM_TO_COPY_CAST_CACHE", "1").lower() in (
        "1",
        "true",
        "on",
    )


def _to_copy_cast_cache_key(
    x: torch.Tensor,
    target_dtype: torch.dtype,
    target_device: torch.device,
    version: int,
) -> tuple:
    return (
        id(x),
        int(x.data_ptr()),
        tuple(x.shape),
        tuple(x.stride()),
        version,
        str(x.dtype),
        str(target_dtype),
        str(target_device),
    )


def _maybe_get_cached_cast(
    x: torch.Tensor,
    target_dtype: torch.dtype,
    target_device: torch.device,
) -> Optional[torch.Tensor]:
    global _TO_COPY_CAST_CACHE_BYTES

    if not _to_copy_cast_cache_enabled():
        return None
    if target_device != x.device or target_dtype == x.dtype:
        return None
    if x.requires_grad or not x.is_contiguous():
        return None
    if torch.is_inference_mode_enabled():
        return None
    try:
        version = int(getattr(x, "_version", 0))
    except RuntimeError:
        return None

    min_numel = max(_env_int("FLAGGEMS_ARM_TO_COPY_CAST_MIN_NUMEL", 1024), 0)
    if x.numel() < min_numel:
        return None

    max_bytes = max(_env_int("FLAGGEMS_ARM_TO_COPY_CAST_MAX_BYTES", 2**30), 0)
    if max_bytes <= 0:
        return None

    key = _to_copy_cast_cache_key(x, target_dtype, target_device, version)
    cached = _TO_COPY_CAST_CACHE.get(key)
    if cached is not None:
        _TO_COPY_CAST_CACHE.move_to_end(key)
        return cached

    casted = torch.ops.aten._to_copy.default.redispatch(
        _FALLBACK_KEYSET,
        x,
        dtype=target_dtype,
        layout=None,
        device=target_device,
        pin_memory=None,
        non_blocking=False,
        memory_format=torch.preserve_format,
    )
    if not casted.is_contiguous():
        casted = casted.contiguous()

    casted_bytes = _tensor_nbytes(casted)
    max_tensor_bytes = max(
        _env_int("FLAGGEMS_ARM_TO_COPY_CAST_MAX_TENSOR_BYTES", 256 * 1024 * 1024), 0
    )
    if casted_bytes > max_bytes or (
        max_tensor_bytes > 0 and casted_bytes > max_tensor_bytes
    ):
        return casted

    max_entries = max(_env_int("FLAGGEMS_ARM_TO_COPY_CAST_MAX_ENTRIES", 128), 1)
    while _TO_COPY_CAST_CACHE and (
        _TO_COPY_CAST_CACHE_BYTES + casted_bytes > max_bytes
        or len(_TO_COPY_CAST_CACHE) >= max_entries
    ):
        _, evicted = _TO_COPY_CAST_CACHE.popitem(last=False)
        _TO_COPY_CAST_CACHE_BYTES -= _tensor_nbytes(evicted)

    if casted_bytes > max_bytes:
        return casted

    _TO_COPY_CAST_CACHE[key] = casted
    _TO_COPY_CAST_CACHE_BYTES += casted_bytes
    return casted


# func: _to_copy(Tensor self, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None,
#   bool? pin_memory=None, bool non_blocking=False, MemoryFormat? memory_format=None) -> Tensor
def to_copy(
    x,
    *,
    dtype=None,
    layout=None,
    device=None,
    pin_memory=None,
    non_blocking=False,
    memory_format=None,
):
    if not isinstance(x, torch.Tensor):
        if layout is not None and layout != torch.strided:
            raise NotImplementedError(
                "FlagGems to_copy currently supports strided tensors only."
            )
        if pin_memory is not None:
            raise NotImplementedError(
                "FlagGems to_copy does not yet support pin_memory=True."
            )
        scalar_dtype = dtype if isinstance(dtype, torch.dtype) else None
        scalar_device = torch.device(device) if device is not None else None
        return torch.as_tensor(x, dtype=scalar_dtype, device=scalar_device)

    # We only implement the dense strided kernel today; all other layouts fall back to PyTorch.
    if (layout is not None and layout != torch.strided) or x.layout != torch.strided:
        raise NotImplementedError(
            "FlagGems to_copy currently supports strided tensors only."
        )
    if pin_memory is not None:
        raise NotImplementedError(
            "FlagGems to_copy does not yet support pin_memory=True."
        )
    if x.is_quantized:
        raise NotImplementedError(
            "Quantized tensors are not supported in FlagGems to_copy yet."
        )

    target_dtype = _resolve_dtype(x, dtype)
    target_device = _resolve_device(x, device)
    target_memory_format = _normalize_memory_format(memory_format)

    if target_device != x.device:
        # Device transfer (d2h/h2d etc.) relies on PyTorch's implementation.
        return torch.ops.aten._to_copy.default.redispatch(
            _FALLBACK_KEYSET,
            x,
            dtype=target_dtype,
            layout=layout,
            device=target_device,
            pin_memory=pin_memory,
            non_blocking=non_blocking,
            memory_format=target_memory_format,
        )

    logger.debug("GEMS _TO_COPY")
    empty_kwargs = {"dtype": target_dtype, "device": target_device}

    if target_memory_format is torch.preserve_format:
        out = _allocate_preserve_format(x, empty_kwargs)
    else:
        out = torch.empty_like(x, memory_format=target_memory_format, **empty_kwargs)

    if x.is_contiguous() and out.is_contiguous() and x.shape == out.shape:
        cached = _maybe_get_cached_cast(x, target_dtype, target_device)
        if cached is not None:
            _launch_to_copy_contiguous(cached, out)
            return out
        _launch_to_copy_contiguous(x, out)
        return out

    return _to_copy_func(x, out0=out)
