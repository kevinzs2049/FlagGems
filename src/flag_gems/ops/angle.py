import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic, tl_extra_shim
from flag_gems.utils.triton_lang_helper import use_tl_extra


@use_tl_extra
@triton.jit
def atan2(y, x):
    # Fallback implementation for backends without libdevice.atan2 (e.g. Triton CPU).
    pi = math.pi
    half_pi = pi * 0.5
    zero = 0.0
    one = 1.0

    safe_x = tl.where(x == zero, one, x)
    base = tl_extra_shim.atan(y / safe_x)

    result = tl.where(x > zero, base, base)
    result = tl.where((x < zero) & (y >= zero), base + pi, result)
    result = tl.where((x < zero) & (y < zero), base - pi, result)
    result = tl.where((x == zero) & (y > zero), half_pi, result)
    result = tl.where((x == zero) & (y < zero), -half_pi, result)
    result = tl.where((x == zero) & (y == zero), zero, result)
    return result

logger = logging.getLogger(__name__)


@pointwise_dynamic(is_tensor=[True, True], promotion_methods=[(0, "DEFAULT")])
@triton.jit
def angle_func(real, imag):
    real_last, imag_last = (
        (real.to(tl.float32), imag.to(tl.float32))
        if real.dtype == tl.float16
        else (real, imag)
    )
    result = atan2(imag_last, real_last)
    return result


@pointwise_dynamic(is_tensor=[True], promotion_methods=[(0, "INT_TO_FLOAT")])
@triton.jit
def angle_float_and_int(real):
    zero = 0.0
    pi = math.pi
    real_positive = real >= zero
    result = tl.where(real_positive, zero, pi)
    return result


def angle(input_tensor: torch.Tensor) -> torch.Tensor:
    logger.debug("GEMS ANGLE")
    if input_tensor.dtype == torch.complex32 or input_tensor.dtype == torch.complex64:
        real = input_tensor.real
        imag = input_tensor.imag
        return angle_func(real, imag)
    else:
        real = input_tensor
        return angle_float_and_int(real)
