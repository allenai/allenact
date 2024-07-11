"""Functions used to initialize and manipulate pytorch models."""
import hashlib
from typing import Sequence, Tuple, Union, Optional, Dict, Any, Callable

import numpy as np
import torch
import torch.nn as nn

from allenact.utils.misc_utils import md5_hash_str_as_int


def md5_hash_of_state_dict(state_dict: Dict[str, Any]):
    hashables = []
    for piece in sorted(state_dict.items()):
        if isinstance(piece[1], (np.ndarray, torch.Tensor, nn.Parameter)):
            hashables.append(piece[0])
            if not isinstance(piece[1], np.ndarray):
                p1 = piece[1].data.cpu().numpy()
            else:
                p1 = piece[1]
            hashables.append(
                int(
                    hashlib.md5(p1.tobytes()).hexdigest(),
                    16,
                )
            )
        else:
            hashables.append(md5_hash_str_as_int(str(piece)))

    return md5_hash_str_as_int(str(hashables))


class Flatten(nn.Module):
    """Flatten input tensor so that it is of shape (FLATTENED_BATCH x -1)."""

    # noinspection PyMethodMayBeStatic
    def forward(self, x):
        """Flatten input tensor.

        # Parameters
        x : Tensor of size (FLATTENED_BATCH x ...) to flatten to size (FLATTENED_BATCH x -1)
        # Returns
        Flattened tensor.
        """
        return x.reshape(x.size(0), -1)


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
    nsteps: int
    nsamplers: int
    nagents: int

    assert len(cnn_input.shape) in [
        5,
        6,
    ], "CNN input must have shape [STEP, SAMPLER, (AGENT,) dim1, dim2, dim3]"

    nagents: Optional[int] = None
    if len(cnn_input.shape) == 6:
        nsteps, nsamplers, nagents = cnn_input.shape[:3]
    else:
        nsteps, nsamplers = cnn_input.shape[:2]

    # Make FLAT_BATCH = nsteps * nsamplers (* nagents)
    cnn_input = cnn_input.view((-1,) + cnn_input.shape[2 + int(nagents is not None) :])

    if permute_order is not None:
        cnn_input = cnn_input.permute(*permute_order)
    cnn_output = cnn(cnn_input)

    if nagents is not None:
        cnn_output = cnn_output.reshape(
            (
                nsteps,
                nsamplers,
                nagents,
            )
            + cnn_output.shape[1:]
        )
    else:
        cnn_output = cnn_output.reshape(
            (
                nsteps,
                nsamplers,
            )
            + cnn_output.shape[1:]
        )

    return cnn_output


def simple_conv_and_linear_weights_init(m):
    if type(m) in [
        nn.Conv1d,
        nn.Conv2d,
        nn.Conv3d,
        nn.ConvTranspose1d,
        nn.ConvTranspose2d,
        nn.ConvTranspose3d,
    ]:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        w_bound = np.sqrt(6.0 / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif type(m) == nn.Linear:
        simple_linear_weights_init(m)


def simple_linear_weights_init(m):
    if type(m) == nn.Linear:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6.0 / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        if m.bias is not None:
            m.bias.data.fill_(0)


class FeatureEmbedding(nn.Module):
    """A wrapper of nn.Embedding but support zero output Used for extracting
    features for actions/rewards."""

    def __init__(self, input_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        if self.output_size != 0:
            self.fc = nn.Embedding(input_size, output_size)
        else:  # automatically be moved to a device
            self.null_embedding: torch.Tensor
            self.register_buffer(
                "null_embedding",
                torch.zeros(
                    0,
                ),
                persistent=False,
            )

    def forward(self, inputs):
        if self.output_size != 0:
            return self.fc(inputs)
        else:
            return self.null_embedding
