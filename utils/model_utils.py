"""Functions used to initialize and manipulate pytorch models."""
from collections import Callable
from typing import Sequence, Tuple, Union, Optional

import torch
from torch import nn


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
    output_height: int,
    output_width: int,
    output_channels: int,
):
    assert (
        len(layer_channels) == len(kernel_sizes) == len(strides)
    ), "Mismatched sizes: layers {} kernels {} strides {}".format(
        layer_channels, kernel_sizes, strides
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
            ),
        )
        if it < len(layer_channels) - 1:
            net.add_module("relu_{}".format(it), nn.ReLU(inplace=True))

    net.add_module("flatten", nn.Flatten())
    net.add_module(
        "fc",
        nn.Linear(layer_channels[-1] * output_width * output_height, output_channels),
    )
    net.add_module("out_relu", nn.ReLU(True))

    return net


def compute_cnn_output(
    cnn: nn.Module,
    cnn_input: torch.Tensor,
    permute_order: Optional[Tuple[int, ...]] = (
        0,
        3,
        1,
        2,
    ),  # [FLAT_BATCH x CHANNEL x HEIGHT X WIDTH] from [FLAT_BATCH x HEIGHT x WIDTH x CHANNEL]
):
    """Computes CNN outputs for given inputs.

    # Parameters

    cnn : A torch CNN.
    cnn_input: A torch Tensor with inputs.
    permute_order: A permutation Tuple to provide PyTorch dimension order, default (0, 3, 1, 2), where 0 corresponds to
                   the flattened batch dimensions (combining batch and step)

    # Returns

    CNN output with dimensions [BATCH (x STEPS) x CHANNEL x HEIGHT X WIDTH].
    """
    nsteps: Optional[int] = None
    bsize: Optional[int] = None
    if len(cnn_input.shape) == 5:
        nsteps, bsize = cnn_input.shape[:2]
        # Make FLAT_BATCH = nsteps * bsize
        cnn_input = cnn_input.reshape((-1) + cnn_input.shape[2:])

    if permute_order is not None:
        cnn_input = cnn_input.permute(*permute_order)
    cnn_output = cnn(cnn_input)

    if nsteps is not None:
        cnn_output = cnn_output.reshape((nsteps, bsize) + cnn_output.shape[1:])

    return cnn_output
