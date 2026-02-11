from backend_utils import VendorInfoBase  # noqa: E402

vendor_info = VendorInfoBase(
    vendor_name="arm", device_name="cpu", device_query_cmd="cat /proc/cpuinfo"
)

# Keep high-traffic CPU primitives on native PyTorch kernels for better end-to-end
# generation throughput on Triton-CPU deployments.
CUSTOMIZED_UNUSED_OPS = (
    "addmm",
    "addmm_out",
    "all",
    "all_dim",
    "all_dims",
    "any",
    "any_dim",
    "any_dims",
    "cat",
    "copy_",
    "cumsum",
    "cumsum_out",
    "fill_scalar",
    "fill_scalar_",
    "fill_tensor",
    "fill_tensor_",
    "isin",
    "lt",
    "lt_scalar",
    "mm",
    "mm_out",
    "mul",
    "mul_",
    "silu",
    "silu_",
    "zero_",
)


__all__ = ["*"]
