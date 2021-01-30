"""Functions used to initialize and manipulate pytorch models."""
from collections import Callable
from typing import Sequence, Tuple, Union, Optional

import numpy as np
import torch


class Flatten(torch.nn.Module):
    """Flatten input tensor so that it is of shape (FLATTENED_BATCH x -1)."""

    def forward(self, x):
        """Flatten input tensor.

        # Parameters
        x : Tensor of size (FLATTENED_BATCH x ...) to flatten to size (FLATTENED_BATCH x -1)
        # Returns
        Flattened tensor.
        """
        return x.reshape(x.size(0), -1)


def init_linear_layer(
    module: torch.nn.Linear, weight_init: Callable, bias_init: Callable, gain=1
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
) -> torch.nn.Module:
    assert (
        len(layer_channels)
        == len(kernel_sizes)
        == len(strides)
        == len(paddings)
        == len(dilations)
    ), "Mismatched sizes: layers {} kernels {} strides {} paddings {} dilations {}".format(
        layer_channels, kernel_sizes, strides, paddings, dilations
    )

    net = torch.nn.Sequential()

    input_channels_list = [input_channels] + list(layer_channels)

    for it, current_channels in enumerate(layer_channels):
        net.add_module(
            "conv_{}".format(it),
            torch.nn.Conv2d(
                in_channels=input_channels_list[it],
                out_channels=current_channels,
                kernel_size=kernel_sizes[it],
                stride=strides[it],
                padding=paddings[it],
                dilation=dilations[it],
            ),
        )
        if it < len(layer_channels) - 1:
            net.add_module("relu_{}".format(it), torch.nn.ReLU(inplace=True))

    if flatten:
        net.add_module("flatten", Flatten())
        net.add_module(
            "fc",
            torch.nn.Linear(
                layer_channels[-1] * output_width * output_height, output_channels
            ),
        )
    if output_relu:
        net.add_module("out_relu", torch.nn.ReLU(True))

    return net


def compute_cnn_output(
    cnn: torch.nn.Module,
    cnn_input: torch.Tensor,
    permute_order: Optional[Tuple[int, ...]] = (
        0,  # FLAT_BATCH (flattening steps, samplers and agents)
        3,  # CHANNEL
        1,  # ROW
        2,  # COL
    ),  # from [FLAT_BATCH x ROW x COL x CHANNEL] flattened input
):
    """Computes CNN outputs for given inputs.

    # Parameters

    cnn : A torch CNN.
    cnn_input: A torch Tensor with inputs.
    permute_order: A permutation Tuple to provide PyTorch dimension order, default (0, 3, 1, 2), where 0 corresponds to
                   the flattened batch dimensions (combining step, sampler and agent)

    # Returns

    CNN output with dimensions [STEP, SAMPLER, AGENT, CHANNEL, (HEIGHT, WIDTH)].
    """
    use_agents: bool
    nsteps: int
    nsamplers: int
    nagents: int

    assert len(cnn_input.shape) in [
        5,
        6,
    ], "CNN input must have shape [STEP, SAMPLER, (AGENT,) dim1, dim2, dim3]"

    if len(cnn_input.shape) == 6:
        nsteps, nsamplers, nagents = cnn_input.shape[:3]
        use_agents = True
    else:
        nsteps, nsamplers = cnn_input.shape[:2]
        nagents = 1
        use_agents = False

    # Make FLAT_BATCH = nsteps * nsamplers (* nagents)
    cnn_input = cnn_input.view((-1,) + cnn_input.shape[2 + int(use_agents) :])

    if permute_order is not None:
        cnn_input = cnn_input.permute(*permute_order)
    cnn_output = cnn(cnn_input)

    if use_agents:
        cnn_output = cnn_output.reshape(
            (nsteps, nsamplers, nagents,) + cnn_output.shape[1:]
        )
    else:
        cnn_output = cnn_output.reshape((nsteps, nsamplers,) + cnn_output.shape[1:])

    return cnn_output


def simple_conv_and_linear_weights_init(m):
    if type(m) in [
        torch.nn.Conv1d,
        torch.nn.Conv2d,
        torch.nn.Conv3d,
        torch.nn.ConvTranspose1d,
        torch.nn.ConvTranspose2d,
        torch.nn.ConvTranspose3d,
    ]:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        w_bound = np.sqrt(6.0 / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif type(m) == torch.nn.Linear:
        simple_linear_weights_init(m)


def simple_linear_weights_init(m):
    if type(m) == torch.nn.Linear:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6.0 / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        if m.bias is not None:
            m.bias.data.fill_(0)
