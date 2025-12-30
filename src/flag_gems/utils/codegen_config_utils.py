import os
from dataclasses import dataclass
from typing import Tuple

import triton

from flag_gems.runtime import device
from flag_gems.runtime.backend import vendor_module
from flag_gems.runtime.commom_utils import vendors


def default_heuristics_for_num_warps(tile_size):
    if tile_size < 2048:
        return 4
    elif tile_size < 4096:
        return 8
    else:
        return 16


def metax_heuristics_for_num_warps(tile_size):
    if tile_size <= 1024:
        return 4
    elif tile_size <= 2048:
        return 8
    else:
        return 16


def cambricon_heuristics_for_num_warps(tile_size):
    return 1


def cpu_heuristics_for_num_warps(tile_size):
    if tile_size <= 512:
        return 2
    return 4

@dataclass
class CodeGenConfig:
    max_tile_size: int
    max_grid_size: Tuple[int, int, int]
    max_num_warps_per_cta: int

    prefer_block_pointer: bool
    prefer_1d_tile: bool
    # gen_configs: -> configs
    # prune_config: (as jit function, ) cofigs -> configs

    def __post_init__(self):
        if self.prefer_1d_tile:
            self.prefer_block_pointer = False


CODEGEN_COFIGS = {
    vendors.NVIDIA: CodeGenConfig(
        512,
        (65536, 65536, 65536),
        32,
        True,
        prefer_1d_tile=int(triton.__version__[0]) < 3,
    ),
    vendors.CAMBRICON: CodeGenConfig(
        8192,
        tuple([vendor_module.TOTAL_CORE_NUM, 1, 1]),
        32,
        False,
        prefer_1d_tile=int(triton.__version__[0]) < 3,
    )
    if vendor_module.vendor_info.vendor_name == "cambricon"
    else None,
    vendors.METAX: CodeGenConfig(
        2048,
        (65536, 65536, 65536),
        16,
        True,
        prefer_1d_tile=int(triton.__version__[0]) < 3,
    ),
    vendors.MTHREADS: CodeGenConfig(
        512,
        (2147483647, 2147483647, 2147483647),
        32,
        True,
        prefer_1d_tile=int(triton.__version__[0]) < 3,
    ),
    vendors.KUNLUNXIN: CodeGenConfig(
        512,
        (65536, 65536, 65536),
        32,
        True,
        prefer_1d_tile=True,
    ),
    vendors.ARM: CodeGenConfig(
        256,
        (os.cpu_count() or 1, 1, 1),
        2,
        False,
        prefer_1d_tile=True,
    ),
}

HEURISTICS_CONFIG = {
    vendors.NVIDIA: default_heuristics_for_num_warps,
    vendors.METAX: metax_heuristics_for_num_warps,
    vendors.CAMBRICON: cambricon_heuristics_for_num_warps,
    vendors.ARM: cpu_heuristics_for_num_warps,
}


def get_codegen_config():
    if device.vendor not in CODEGEN_COFIGS:
        return CODEGEN_COFIGS.get(vendors.NVIDIA)
    return CODEGEN_COFIGS.get(device.vendor)


def get_heuristics_for_num_warps(tile_size):
    config = get_codegen_config() or CODEGEN_COFIGS.get(vendors.NVIDIA)
    heuristics_fn = HEURISTICS_CONFIG.get(
        device.vendor, HEURISTICS_CONFIG.get(vendors.NVIDIA)
    )
    num_warps = heuristics_fn(tile_size)
    if config is not None:
        num_warps = min(num_warps, config.max_num_warps_per_cta)
    return max(1, num_warps)
