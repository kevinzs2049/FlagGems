from backend_utils import VendorInfoBase  # noqa: E402

vendor_info = VendorInfoBase(
    vendor_name="arm", device_name="cpu", device_query_cmd="cat /proc/cpuinfo"
)

# Ops listed here are excluded from the ARM customized registry.
CUSTOMIZED_UNUSED_OPS = (
    "all",
    "all_dim",
    "all_dims",
    "any",
    "any_dim",
    "any_dims",
    "bitwise_and_scalar",
    "bitwise_and_scalar_",
    "bitwise_and_scalar_tensor",
    "bitwise_and_tensor",
    "bitwise_and_tensor_",
    "bitwise_or_scalar",
    "bitwise_or_scalar_",
    "bitwise_or_scalar_tensor",
    "bitwise_or_tensor",
    "bitwise_or_tensor_",
    "bitwise_not",
    "bitwise_not_",
    "cat",
    "copy_",
    "cumsum",
    "cumsum_out",
    "fill_scalar",
    "fill_scalar_",
    "fill_tensor",
    "fill_tensor_",
    "eq",
    "eq_scalar",
    "ge",
    "ge_scalar",
    "isin",
    "lt",
    "lt_scalar",
    "silu",
    "silu_",
    "zero_",
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
    # Attention ops with autotune cause non-deterministic behavior on CPU
    "scaled_dot_product_attention",
    "scaled_dot_product_attention_forward",
    "scaled_dot_product_attention_backward",
)


__all__ = ["*"]
