"""FlagGems implementation of aten::linear.

Directly intercepts aten::linear(input, weight, bias?) to avoid the
CompositeImplicitAutograd decomposition overhead (t + mm/addmm).
Delegates to the already-optimized mm/addmm implementations which
have ARM-specific M=1 decode fastpaths.

For small output dimensions (N < 256), delegates to mm/addmm directly
which have their own small-shape handling.
"""

import logging

import torch

logger = logging.getLogger(__name__)


def linear(input, weight, bias=None):
    """aten::linear(Tensor input, Tensor weight, Tensor? bias=None) -> Tensor

    Computes input @ weight.T [+ bias].
    For 2D input: delegates to mm (no bias) or addmm (with bias).
    For batched input (3D+): reshapes to 2D, computes, reshapes back.
    """
    weight_t = weight.t()

    if input.dim() == 2:
        if bias is not None:
            return torch.addmm(bias, input, weight_t)
        else:
            return torch.mm(input, weight_t)
    else:
        orig_shape = input.shape
        input_2d = input.reshape(-1, orig_shape[-1])
        if bias is not None:
            output_2d = torch.addmm(bias, input_2d, weight_t)
        else:
            output_2d = torch.mm(input_2d, weight_t)
        output_shape = orig_shape[:-1] + (weight.shape[0],)
        return output_2d.reshape(output_shape)
