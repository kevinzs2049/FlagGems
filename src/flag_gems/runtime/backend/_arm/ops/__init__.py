from .add import add, add_
from .addmm import addmm, addmm_out
from .all import all
from .any import any
from .arange import arange
from .argmax import argmax
from .attention import scaled_dot_product_attention
from .bmm import bmm
from .cat import cat
from .cos import cos
from .cumsum import cumsum
from .div import (
    div_mode,
    div_mode_,
    floor_divide,
    floor_divide_,
    remainder,
    remainder_,
    true_divide,
    true_divide_,
)
from .embedding import embedding
from .exponential_ import exponential_
from .full import full
from .gather import gather
from .gelu import gelu
from .index_select import index_select
from .index import index
from .isin import isin
from .log_softmax import log_softmax
from .lt import lt
from .masked_fill import masked_fill
from .max import max
from .mean import mean, mean_dim
from .min import min
from .mm import mm, mm_out
from .mul import mul, mul_
from .multinomial import multinomial
from .neg import neg, neg_
from .ones import ones
from .ones_like import ones_like
from .pow import (
    pow_scalar,
    pow_tensor_scalar,
    pow_tensor_scalar_,
    pow_tensor_tensor,
    pow_tensor_tensor_,
)
from .quantile import quantile
from .rsqrt import rsqrt, rsqrt_
from .scatter import scatter
from .silu import silu
from .sin import sin
from .softmax import softmax
from .sort import sort
from .sub import sub
from .sum import sum
from .topk import topk
from .where import where_self_out
from .zeros import zeros

__all__ = [
    "add",
    "add_",
    "addmm",
    "addmm_out",
    "all",
    "any",
    "arange",
    "argmax",
    "bmm",
    "cat",
    "cos",
    "cumsum",
    "div_mode",
    "div_mode_",
    "embedding",
    "exponential_",
    "floor_divide",
    "floor_divide_",
    "full",
    "gather",
    "gelu",
    "index_select",
    "index",
    "isin",
    "log_softmax",
    "lt",
    "masked_fill",
    "max",
    "mean",
    "mean_dim",
    "min",
    "mm",
    "mm_out",
    "mul",
    "mul_",
    "multinomial",
    "neg",
    "neg_",
    "ones",
    "ones_like",
    "pow_scalar",
    "pow_tensor_scalar",
    "pow_tensor_scalar_",
    "pow_tensor_tensor",
    "pow_tensor_tensor_",
    "quantile",
    "remainder",
    "remainder_",
    "rsqrt",
    "rsqrt_",
    "scaled_dot_product_attention",
    "scatter",
    "silu",
    "sin",
    "softmax",
    "sort",
    "sub",
    "sum",
    "topk",
    "true_divide",
    "true_divide_",
    "where_self_out",
    "zeros",
]

# Register Triton-CPU INT8 GEMM for quantized::linear_dynamic (quantized:: namespace,
# not aten::, so handled separately from the main FlagGems aten_lib registrations).
from .quantized_linear_dynamic import register as _register_quantized_linear_dynamic

_register_quantized_linear_dynamic()

# Register Triton-CPU INT8 GEMM for aten::_int_mm (enables torchao INT8 paths).
from .int_mm import register as _register_int_mm

_register_int_mm()

# Register FlagGems argmax for aten::argmax (decode lm_head: 2.2x faster for [1,151936]).
# Auto-registered on import so INT8 users get the speedup without explicit only_enable().
import logging as _logging
import torch as _torch
from .argmax import argmax as _fg_argmax

_argmax_aten_lib = None


def _register_argmax():
    global _argmax_aten_lib
    if _argmax_aten_lib is not None:
        return
    try:
        _argmax_aten_lib = _torch.library.Library("aten", "IMPL")
        _argmax_aten_lib.impl("argmax", _fg_argmax, "CPU", allow_override=True)
        _logging.getLogger(__name__).debug(
            "FlagGems ARM: registered Triton-CPU argmax for aten::argmax"
        )
    except Exception as e:
        _logging.getLogger(__name__).warning(
            f"FlagGems ARM: failed to register argmax override: {e}"
        )


_register_argmax()
