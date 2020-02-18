"""Basic building block torch networks that can be used across a variety of
tasks."""

from typing import Sequence, Tuple, Dict, Union, cast, List

import numpy as np
import torch
from torch import nn as nn
import torchvision.models as models
from gym.spaces.dict import Dict as SpaceDict


class Flatten(nn.Module):
    """Flatten input tensor so that it is of shape (batchs x -1)."""

    def forward(self, x):
        """Flatten input tensor.
        # Parameters
        x : Tensor of size (batches x ...) to flatten to size (batches x -1)
        # Returns
        Flattened tensor.
        """
        return x.view(x.size(0), -1)


class SimpleCNN(nn.Module):
    """A Simple 3-Conv CNN followed by a fully connected layer.
    Takes in observations (of type gym.spaces.dict) and produces an embedding
     of the `"rgb"` and/or `"depth"` components.
    # Attributes
    observation_space : The observation_space of the agent, should have 'rgb' or 'depth' as
        a component (otherwise it is a blind model).
    output_size : The size of the embedding vector to produce.
    """

    def __init__(self, observation_space: SpaceDict, output_size: int):
        """Initializer.
        # Parameters
        observation_space : See class attributes documentation.
        output_size : See class attributes documentation.
        """
        super().__init__()
        if "rgb" in observation_space.spaces:
            self._n_input_rgb = observation_space.spaces["rgb"].shape[2]
        else:
            self._n_input_rgb = 0

        if "depth" in observation_space.spaces:
            self._n_input_depth = observation_space.spaces["depth"].shape[2]
        else:
            self._n_input_depth = 0

        # kernel size for different CNN layers
        self._cnn_layers_kernel_size = [(8, 8), (4, 4), (3, 3)]

        # strides for different CNN layers
        self._cnn_layers_stride = [(4, 4), (2, 2), (1, 1)]

        if self._n_input_rgb > 0:
            cnn_dims = np.array(
                observation_space.spaces["rgb"].shape[:2], dtype=np.float32
            )
        elif self._n_input_depth > 0:
            cnn_dims = np.array(
                observation_space.spaces["depth"].shape[:2], dtype=np.float32
            )
        else:
            assert self.is_blind

        if self.is_blind:
            self.cnn = nn.Sequential()
        else:
            for kernel_size, stride in zip(
                self._cnn_layers_kernel_size, self._cnn_layers_stride
            ):
                # noinspection PyUnboundLocalVariable
                cnn_dims = self._conv_output_dim(
                    dimension=cnn_dims,
                    padding=np.array([0, 0], dtype=np.float32),
                    dilation=np.array([1, 1], dtype=np.float32),
                    kernel_size=np.array(kernel_size, dtype=np.float32),
                    stride=np.array(stride, dtype=np.float32),
                )

            # noinspection PyUnboundLocalVariable
            self.cnn = nn.Sequential(
                nn.Conv2d(
                    in_channels=self._n_input_rgb + self._n_input_depth,
                    out_channels=32,
                    kernel_size=self._cnn_layers_kernel_size[0],
                    stride=self._cnn_layers_stride[0],
                ),
                nn.ReLU(True),
                nn.Conv2d(
                    in_channels=32,
                    out_channels=64,
                    kernel_size=self._cnn_layers_kernel_size[1],
                    stride=self._cnn_layers_stride[1],
                ),
                nn.ReLU(True),
                nn.Conv2d(
                    in_channels=64,
                    out_channels=32,
                    kernel_size=self._cnn_layers_kernel_size[2],
                    stride=self._cnn_layers_stride[2],
                ),
                #  nn.ReLU(True),
                Flatten(),
                nn.Linear(32 * cnn_dims[0] * cnn_dims[1], output_size),
                nn.ReLU(True),
            )

        self.layer_init()

    @staticmethod
    def _conv_output_dim(
        dimension: Sequence[int],
        padding: Sequence[int],
        dilation: Sequence[int],
        kernel_size: Sequence[int],
        stride: Sequence[int],
    ) -> Tuple[int, ...]:
        """Calculates the output height and width based on the input height and
        width to the convolution layer.
        For parameter definitions see [here](https://pytorch.org/docs/master/nn.html#torch.nn.Conv2d).
        # Parameters
        dimension : See above link.
        padding : See above link.
        dilation : See above link.
        kernel_size : See above link.
        stride : See above link.
        """
        assert len(dimension) == 2
        out_dimension = []
        for i in range(len(dimension)):
            out_dimension.append(
                int(
                    np.floor(
                        (
                            (
                                dimension[i]
                                + 2 * padding[i]
                                - dilation[i] * (kernel_size[i] - 1)
                                - 1
                            )
                            / stride[i]
                        )
                        + 1
                    )
                )
            )
        return tuple(out_dimension)

    def layer_init(self) -> None:
        """Initialize layer parameters using kaiming normal."""
        for layer in self.cnn:
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(layer.weight, nn.init.calculate_gain("relu"))
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, val=0)

    @property
    def is_blind(self):
        """True if the observation space doesn't include `"rgb"` or
        `"depth"`."""
        return self._n_input_rgb + self._n_input_depth == 0

    def forward(self, observations: Dict[str, torch.Tensor]):
        cnn_input_list = []
        if self._n_input_rgb > 0:
            rgb_observations = observations["rgb"]
            # permute tensor to dimension [BATCH x CHANNEL x HEIGHT X WIDTH]
            rgb_observations = rgb_observations.permute(0, 3, 1, 2)
            # rgb_observations = rgb_observations / 255.0  # normalize RGB
            cnn_input_list.append(rgb_observations)

        if self._n_input_depth > 0:
            depth_observations = observations["depth"]
            # permute tensor to dimension [BATCH x CHANNEL x HEIGHT X WIDTH]
            depth_observations = depth_observations.permute(0, 3, 1, 2)
            cnn_input_list.append(depth_observations)

        cnn_input = torch.cat(cnn_input_list, dim=1)

        return self.cnn(cnn_input)


class ResNet50(nn.Module):

    def __init__(self, observation_space: SpaceDict, output_size: int, pretrained=True):
        """Initializer.
        # Parameters
        observation_space : See class attributes documentation.
        output_size : See class attributes documentation.
        """
        super().__init__()
        if "rgb" in observation_space.spaces:
            self._n_input_rgb = observation_space.spaces["rgb"].shape[2]
        else:
            self._n_input_rgb = 0

        if "depth" in observation_space.spaces:
            self._n_input_depth = observation_space.spaces["depth"].shape[2]
        else:
            self._n_input_depth = 0

        if self._n_input_rgb > 0:
            cnn_dims = np.array(
                observation_space.spaces["rgb"].shape[:2], dtype=np.float32
            )
        elif self._n_input_depth > 0:
            cnn_dims = np.array(
                observation_space.spaces["depth"].shape[:2], dtype=np.float32
            )
        else:
            assert self.is_blind

        if self.is_blind:
            self.cnn = nn.Sequential()
        else:
            # Initialized pre trained ResNet50 with the last linear layer cut off
            # Add a untrained linear layer mapping from 2048 ResNet feature space to output_size dimensions
            self.cnn = nn.Sequential(
                *list(models.resnet50(pretrained=pretrained).children())[:-1] +
                [nn.Flatten(), nn.Linear(2048, output_size)]
            )

    @property
    def is_blind(self):
        """True if the observation space doesn't include `"rgb"` or
        `"depth"`."""
        return self._n_input_rgb + self._n_input_depth == 0

    def forward(self, observations: Dict[str, torch.Tensor]):
        cnn_input_list = []
        if self._n_input_rgb > 0:
            rgb_observations = observations["rgb"]
            # permute tensor to dimension [BATCH x CHANNEL x HEIGHT X WIDTH]
            rgb_observations = rgb_observations.permute(0, 3, 1, 2)
            # rgb_observations = rgb_observations / 255.0  # normalize RGB
            cnn_input_list.append(rgb_observations)

        if self._n_input_depth > 0:
            depth_observations = observations["depth"]
            # permute tensor to dimension [BATCH x CHANNEL x HEIGHT X WIDTH]
            depth_observations = depth_observations.permute(0, 3, 1, 2)
            cnn_input_list.append(depth_observations)

        cnn_input = torch.cat(cnn_input_list, dim=1)

        return self.cnn(cnn_input)


class RNNStateEncoder(nn.Module):
    """A simple RNN-based model playing a role in many baseline embodied-
    navigation agents.
    See `seq_forward` for more details of how this model is used.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        rnn_type: str = "GRU",
        trainable_masked_hidden_state: bool = False,
    ):
        """An RNN for encoding the state in RL.
        Supports masking the hidden state during various timesteps in the forward lass
        # Parameters
        input_size : The input size of the RNN.
        hidden_size : The hidden size.
        num_layers : The number of recurrent layers.
        rnn_type : The RNN cell type.  Must be GRU or LSTM.
        trainable_masked_hidden_state : If `True` the initial hidden state (used at the start of a Task)
            is trainable (as opposed to being a vector of zeros).
        """

        super().__init__()
        self._num_recurrent_layers = num_layers
        self._rnn_type = rnn_type

        self.rnn = getattr(nn, rnn_type)(
            input_size=input_size, hidden_size=hidden_size, num_layers=num_layers
        )

        self.trainable_masked_hidden_state = trainable_masked_hidden_state
        if trainable_masked_hidden_state:
            self.init_hidden_state = nn.Parameter(
                0.1 * torch.randn((num_layers, 1, hidden_size)), requires_grad=True
            )

        self.layer_init()

    def layer_init(self):
        """Initialize the RNN parameters in the model."""
        for name, param in self.rnn.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.constant_(param, 0)

    @property
    def num_recurrent_layers(self) -> int:
        """The number of recurrent layers in the network."""
        return self._num_recurrent_layers * (2 if "LSTM" in self._rnn_type else 1)

    def _pack_hidden(
        self, hidden_states: Union[torch.FloatTensor, Sequence[torch.FloatTensor]]
    ) -> torch.FloatTensor:
        """Stacks hiddens states in an LSTM together (if using a GRU rather
        than an LSTM this is just the identitiy).
        # Parameters
        hidden_states : The hidden states to (possibly) stack.
        """
        if "LSTM" in self._rnn_type:
            hidden_states = cast(
                torch.FloatTensor,
                torch.cat([hidden_states[0], hidden_states[1]], dim=0),
            )

        return cast(torch.FloatTensor, hidden_states)

    def _unpack_hidden(
        self, hidden_states: torch.FloatTensor
    ) -> Union[torch.FloatTensor, Tuple[torch.FloatTensor, torch.FloatTensor]]:
        """Partial inverse of `_pack_hidden` (exact if there are 2 hidden
        layers)."""
        if "LSTM" in self._rnn_type:
            new_hidden_states = (
                hidden_states[0 : self._num_recurrent_layers],
                hidden_states[self._num_recurrent_layers :],
            )
            return cast(Tuple[torch.FloatTensor, torch.FloatTensor], new_hidden_states)
        return hidden_states

    def _mask_hidden(
        self,
        hidden_states: Union[Tuple[torch.FloatTensor, ...], torch.FloatTensor],
        masks: torch.FloatTensor,
    ) -> Union[Tuple[torch.FloatTensor, ...], torch.FloatTensor]:
        """Mask input hidden states given `masks`.
        Useful when masks represent steps on which a task has completed.
        # Parameters
        hidden_states : The hidden states.
        masks : Masks to apply to hidden states (see seq_forward).
        # Returns
        Masked hidden states. Here masked hidden states will be replaced with
        either all zeros (if `trainable_masked_hidden_state` was False) and will
        otherwise be a learnable collection of parameters.
        """
        if not self.trainable_masked_hidden_state:
            if isinstance(hidden_states, tuple):
                hidden_states = tuple(
                    cast(torch.FloatTensor, v * masks) for v in hidden_states
                )
            else:
                hidden_states = cast(torch.FloatTensor, masks * hidden_states)
        else:
            if isinstance(hidden_states, tuple):
                # noinspection PyTypeChecker
                hidden_states = tuple(
                    v * masks
                    + (1.0 - masks) * (self.init_hidden_state.repeat(1, v.shape[1], 1))  # type: ignore
                    for v in hidden_states
                )  # type: ignore
            else:
                # noinspection PyTypeChecker
                hidden_states = masks * hidden_states + (1 - masks) * (  # type: ignore
                    self.init_hidden_state.repeat(1, hidden_states.shape[1], 1)
                )

        return hidden_states

    def single_forward(
        self,
        x: torch.FloatTensor,
        hidden_states: torch.FloatTensor,
        masks: torch.FloatTensor,
    ) -> Tuple[
        torch.FloatTensor, Union[torch.FloatTensor, Tuple[torch.FloatTensor, ...]]
    ]:
        """Forward for a non-sequence input."""
        unpacked_hidden_states = self._unpack_hidden(hidden_states)
        x, unpacked_hidden_states = self.rnn(
            x.unsqueeze(0),
            self._mask_hidden(
                unpacked_hidden_states, cast(torch.FloatTensor, masks.unsqueeze(0))
            ),
        )
        x = cast(torch.FloatTensor, x.squeeze(0))
        hidden_states = self._pack_hidden(unpacked_hidden_states)
        return x, hidden_states

    def seq_forward(
        self,
        x: torch.FloatTensor,
        hidden_states: torch.FloatTensor,
        masks: torch.FloatTensor,
    ) -> Tuple[
        torch.FloatTensor, Union[torch.FloatTensor, Tuple[torch.FloatTensor, ...]]
    ]:
        """Forward for a sequence of length T.
        # Parameters
        x : (T, N, -1) Tensor that has been flattened to (T * N, -1).
        hidden_states : The starting hidden states.
        masks : A (T, N) tensor flattened to (T * N).
            The masks to be applied to hidden state at every timestep, equal to 0 whenever the previous step finalized
            the task, 1 elsewhere.
        """
        # x is a (T, N, -1) tensor flattened to (T * N, -1)
        n = hidden_states.size(1)
        t = int(x.size(0) / n)

        # unflatten
        x = cast(torch.FloatTensor, x.view(t, n, x.size(1)))
        masks = cast(torch.FloatTensor, masks.view(t, n))

        # steps in sequence which have zero for any agent. Assume t=0 has
        # a zero in it.
        has_zeros = (masks[1:] == 0.0).any(dim=-1).nonzero().squeeze().cpu()

        # +1 to correct the masks[1:]
        if has_zeros.dim() == 0:
            # handle scalar
            has_zeros = [has_zeros.item() + 1]  # type: ignore
        else:
            has_zeros = (has_zeros + 1).numpy().tolist()

        # add t=0 and t=T to the list
        has_zeros = [0] + has_zeros + [t]

        unpacked_hidden_states = self._unpack_hidden(hidden_states)
        outputs = []
        for i in range(len(cast(List, has_zeros)) - 1):
            # process steps that don't have any zeros in masks together
            start_idx = int(has_zeros[i])
            end_idx = int(has_zeros[i + 1])

            # noinspection PyTypeChecker
            rnn_scores, unpacked_hidden_states = self.rnn(
                x[start_idx:end_idx],
                self._mask_hidden(
                    unpacked_hidden_states,
                    cast(torch.FloatTensor, masks[start_idx].view(1, -1, 1)),
                ),
            )

            outputs.append(rnn_scores)

        # x is a (T, N, -1) tensor
        x = cast(torch.FloatTensor, torch.cat(outputs, dim=0).view(t * n, -1))
        hidden_states = self._pack_hidden(unpacked_hidden_states)
        return x, hidden_states

    def forward(
        self,
        x: torch.FloatTensor,
        hidden_states: torch.FloatTensor,
        masks: torch.FloatTensor,
    ) -> Tuple[
        torch.FloatTensor, Union[torch.FloatTensor, Tuple[torch.FloatTensor, ...]]
    ]:
        """Calls `seq_forward` or `single_forward` depending on the input size.
        See the above methods for more information.
        """
        if x.size(0) == hidden_states.size(1):
            return self.single_forward(x, hidden_states, masks)
        else:
            return self.seq_forward(x, hidden_states, masks)


class AddBias(nn.Module):
    """Adding bias parameters to input values."""

    def __init__(self, bias: torch.FloatTensor):
        """Initializer.
        # Parameters
        bias : data to use as the initial values of the bias.
        """
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """Adds the stored bias parameters to `x`."""
        assert x.dim() in [2, 4]

        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        else:
            bias = self._bias.t().view(1, -1, 1, 1)

        return x + bias
