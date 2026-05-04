"""ARM CPU INT4 (W4A8 per-channel) model utilities.

Drop-in nn.Linear replacement using TLE SDOT W4A8 GEMV for decode (T=1)
and on-the-fly i4→i8 unpack + torch._int_mm for prefill (T>1).
"""

from .tle_int4_linear import TLEInt4Linear, pack_w4_per_channel  # noqa: F401
from .quantize_live_w4 import quantize_w4_and_replace_linears  # noqa: F401

__all__ = [
    "TLEInt4Linear",
    "pack_w4_per_channel",
    "quantize_w4_and_replace_linears",
]
