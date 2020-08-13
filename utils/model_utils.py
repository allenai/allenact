"""Functions used to initialize and manipulate pytorch models."""
from collections import Callable
from typing import Sequence, Tuple, Union, Optional

import torch
from torch import nn


class Flatten(nn.Module):
    """Flatten input tensor so that it is of shape (steps x samplers x agents x -1)."""

    def forward(self, x):
        """Flatten input tensor.

        # Parameters
        x : Tensor of size (steps x samplers x agents x ...) to flatten to size (steps x samplers x agents x -1)
        # Returns
        Flattened tensor.
        """
        return x.reshape(x.size(0), x.size(1), x.size(2), -1)


def init_linear_layer(
    module: nn.Linear, weight_init: Callable, bias_init: Callable, gain=1
):
    """Initialize a torch.nn.Linear layer.

    # Parameters

    module : A torch linear layer.
    weight_init : Function used to initialize the weight parameters of the linear layer. Should take the weight data
        tensor and gain as input.
    bias_init : Function used to initialize the bias parameters of the linear layer. Should take the bias data
        tensor and gain as input.
    gain : The gain to apply.

    # Returns

    The initialized linear layer.
    """
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


def grad_norm(parameters, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    if norm_type == "inf":
        total_norm = max(p.grad.data.abs().max() for p in parameters)
    else:
        total_norm = 0
        for p in parameters:
            param_norm = p.grad.data.norm(norm_type)
            total_norm += param_norm.item() ** norm_type
        total_norm = total_norm ** (1.0 / norm_type)
    return total_norm


def make_cnn(
    input_channels: int,
    layer_channels: Sequence[int],
    kernel_sizes: Sequence[Union[int, Tuple[int, int]]],
    strides: Sequence[Union[int, Tuple[int, int]]],
    paddings: Sequence[Union[int, Tuple[int, int]]],
    dilations: Sequence[Union[int, Tuple[int, int]]],
    output_height: int,
    output_width: int,
    output_channels: int,
    flatten: bool = True,
    output_relu: bool = True,
) -> nn.Module:
    assert (
        len(layer_channels)
        == len(kernel_sizes)
        == len(strides)
        == len(paddings)
        == len(dilations)
    ), "Mismatched sizes: layers {} kernels {} strides {} paddings {} dilations {}".format(
        layer_channels, kernel_sizes, strides, paddings, dilations
    )

    net = nn.Sequential()

    input_channels_list = [input_channels] + list(layer_channels)

    for it, current_channels in enumerate(layer_channels):
        net.add_module(
            "conv_{}".format(it),
            nn.Conv2d(
                in_channels=input_channels_list[it],
                out_channels=current_channels,
                kernel_size=kernel_sizes[it],
                stride=strides[it],
                padding=paddings[it],
                dilation=dilations[it],
            ),
        )
        if it < len(layer_channels) - 1:
            net.add_module("relu_{}".format(it), nn.ReLU(inplace=True))

    if flatten:
        net.add_module("flatten", Flatten())
        net.add_module(
            "fc",
            nn.Linear(
                layer_channels[-1] * output_width * output_height, output_channels
            ),
        )
    if output_relu:
        net.add_module("out_relu", nn.ReLU(True))

    return net


def compute_cnn_output(
    cnn: nn.Module,
    cnn_input: torch.Tensor,
    permute_order: Optional[Tuple[int, ...]] = (
        0,  # FLAT_BATCH (flattening steps, samplers and agents)
        3,  # CHANNELS
        1,  # HEIGHT
        2,  # WIDTH
    ),  # from [FLAT_BATCH x HEIGHT x WIDTH x CHANNEL] flattened input
):
    """Computes CNN outputs for given inputs.

    # Parameters

    cnn : A torch CNN.
    cnn_input: A torch Tensor with inputs.
    permute_order: A permutation Tuple to provide PyTorch dimension order, default (0, 3, 1, 2), where 0 corresponds to
                   the flattened batch dimensions (combining step, sampler and agent)

    # Returns

    CNN output with dimensions [STEPS x SAMPLER x AGENT x CHANNEL (x HEIGHT X WIDTH)].
    """
    nsteps, nsamplers, nagents = cnn_input.shape[:3]
    # Make FLAT_BATCH = nsteps * nsamplers * nagents
    cnn_input = cnn_input.reshape((-1,) + cnn_input.shape[3:])

    if permute_order is not None:
        cnn_input = cnn_input.permute(*permute_order)
    cnn_output = cnn(cnn_input)

    cnn_output = cnn_output.reshape(
        (nsteps, nsamplers, nagents,) + cnn_output.shape[1:]
    )

    return cnn_output
