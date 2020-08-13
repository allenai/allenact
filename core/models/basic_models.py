"""Basic building block torch networks that can be used across a variety of
tasks."""
from typing import (
    Sequence,
    Dict,
    Union,
    cast,
    List,
    Callable,
    Optional,
    Tuple,
    Any,
)

import gym
import numpy as np
import torch
from gym.spaces.dict import Dict as SpaceDict
from torch import nn

from core.algorithms.onpolicy_sync.policy import ActorCriticModel, DistributionType
from core.base_abstractions.misc import ActorCriticOutput
from core.base_abstractions.distributions import CategoricalDistr
from utils.model_utils import make_cnn, compute_cnn_output, Flatten


class SimpleCNN(nn.Module):
    """A Simple 3-Conv CNN followed by a fully connected layer. Takes in
    observations (of type gym.spaces.dict) and produces an embedding of the
    `"rgb"` and/or `"depth"` components.

    # Attributes

    observation_space : The observation_space of the agent, should have 'rgb' or 'depth' as
        a component (otherwise it is a blind model).
    output_size : The size of the embedding vector to produce.
    """

    def __init__(
        self,
        observation_space: SpaceDict,
        output_size: int,
        layer_channels: Sequence[int] = (32, 64, 32),
        kernel_sizes: Sequence[Tuple[int, int]] = ((8, 8), (4, 4), (3, 3)),
        layers_stride: Sequence[Tuple[int, int]] = ((4, 4), (2, 2), (1, 1)),
        paddings: Sequence[Tuple[int, int]] = ((0, 0), (0, 0), (0, 0)),
        dilations: Sequence[Tuple[int, int]] = ((1, 1), (1, 1), (1, 1)),
        rgb_uuid: str = "rgb",
        depth_uuid: str = "depth",
    ):
        """Initializer.

        # Parameters

        observation_space : See class attributes documentation.
        output_size : See class attributes documentation.
        """
        super().__init__()

        self.rgb_uuid = rgb_uuid
        self.depth_uuid = depth_uuid

        if self.rgb_uuid in observation_space.spaces:
            self._n_input_rgb = observation_space.spaces[self.rgb_uuid].shape[2]
        else:
            self._n_input_rgb = 0

        if self.depth_uuid in observation_space.spaces:
            self._n_input_depth = observation_space.spaces[self.depth_uuid].shape[2]
        else:
            self._n_input_depth = 0

        # layer hyperparameters
        self._cnn_layers_channels = list(layer_channels)
        self._cnn_layers_kernel_size = list(kernel_sizes)
        self._cnn_layers_stride = list(layers_stride)
        self._cnn_layers_paddings = list(paddings)
        self._cnn_layers_dilations = list(dilations)

        if self._n_input_rgb > 0:
            rgb_cnn_dims = np.array(
                observation_space.spaces[self.rgb_uuid].shape[:2], dtype=np.float32
            )
        if self._n_input_depth > 0:
            depth_cnn_dims = np.array(
                observation_space.spaces[self.depth_uuid].shape[:2], dtype=np.float32
            )
        if self._n_input_rgb <= 0 and self._n_input_depth <= 0:
            assert self.is_blind

        if self.is_blind:
            self.cnn = nn.Sequential()
        else:
            if self._n_input_rgb > 0:
                for kernel_size, stride, padding, dilation in zip(
                    self._cnn_layers_kernel_size,
                    self._cnn_layers_stride,
                    self._cnn_layers_paddings,
                    self._cnn_layers_dilations,
                ):
                    # noinspection PyUnboundLocalVariable
                    rgb_cnn_dims = self._conv_output_dim(
                        dimension=rgb_cnn_dims,
                        padding=np.array(padding, dtype=np.float32),
                        dilation=np.array(dilation, dtype=np.float32),
                        kernel_size=np.array(kernel_size, dtype=np.float32),
                        stride=np.array(stride, dtype=np.float32),
                    )
                # noinspection PyUnboundLocalVariable
                self.rgb_cnn = make_cnn(
                    input_channels=self._n_input_rgb,
                    layer_channels=self._cnn_layers_channels,
                    kernel_sizes=self._cnn_layers_kernel_size,
                    strides=self._cnn_layers_stride,
                    paddings=self._cnn_layers_paddings,
                    dilations=self._cnn_layers_dilations,
                    output_height=rgb_cnn_dims[0],
                    output_width=rgb_cnn_dims[1],
                    output_channels=output_size,
                )
                self.layer_init(self.rgb_cnn)

            if self._n_input_depth > 0:
                for kernel_size, stride, padding, dilation in zip(
                    self._cnn_layers_kernel_size,
                    self._cnn_layers_stride,
                    self._cnn_layers_paddings,
                    self._cnn_layers_dilations,
                ):
                    # noinspection PyUnboundLocalVariable
                    depth_cnn_dims = self._conv_output_dim(
                        dimension=depth_cnn_dims,
                        padding=np.array(padding, dtype=np.float32),
                        dilation=np.array(dilation, dtype=np.float32),
                        kernel_size=np.array(kernel_size, dtype=np.float32),
                        stride=np.array(stride, dtype=np.float32),
                    )
                # noinspection PyUnboundLocalVariable
                self.rgb_cnn = make_cnn(
                    input_channels=self._n_input_depth,
                    layer_channels=self._cnn_layers_channels,
                    kernel_sizes=self._cnn_layers_kernel_size,
                    strides=self._cnn_layers_stride,
                    paddings=self._cnn_layers_paddings,
                    dilations=self._cnn_layers_dilations,
                    output_height=depth_cnn_dims[0],
                    output_width=depth_cnn_dims[1],
                    output_channels=output_size,
                )
                self.layer_init(self.depth_cnn)

    @staticmethod
    def _conv_output_dim(
        dimension: Sequence[int],
        padding: Sequence[int],
        dilation: Sequence[int],
        kernel_size: Sequence[int],
        stride: Sequence[int],
    ) -> Tuple[int, ...]:
        """Calculates the output height and width based on the input height and
        width to the convolution layer. For parameter definitions see.

        [here](https://pytorch.org/docs/master/nn.html#torch.nn.Conv2d).

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

    @staticmethod
    def layer_init(cnn) -> None:
        """Initialize layer parameters using Kaiming normal."""
        for layer in cnn:
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(layer.weight, nn.init.calculate_gain("relu"))
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, val=0)

    @property
    def is_blind(self):
        """True if the observation space doesn't include `"rgb"` or
        `"depth"`."""
        return self._n_input_rgb + self._n_input_depth == 0

    def forward(self, observations: Dict[str, torch.Tensor]):  # type: ignore
        cnn_output_list: List[torch.Tensor] = []

        if self.is_blind:
            return torch.cat(cnn_output_list, dim=1)

        if self._n_input_rgb > 0:
            cnn_output_list.append(
                compute_cnn_output(self.rgb_cnn, observations["rgb"])
            )

        if self._n_input_depth > 0:
            cnn_output_list.append(
                compute_cnn_output(self.depth_cnn, observations["depth"])
            )

        if len(cnn_output_list[0].shape) == 2:
            channels_dim = 1  # for [sampler x channels] (flattened)
        elif len(cnn_output_list[0].shape) == 3:
            channels_dim = 2  # for [step x sampler x channels] (flattened)
        elif len(cnn_output_list[0].shape) == 4:
            channels_dim = 1  # for [sampler x channel x height x width]
        elif len(cnn_output_list[0].shape) == 5:
            channels_dim = 2  # [step x sampler x channel x height x width]
        else:
            raise RuntimeError(
                "Wrong CNN output tensor shape {}".format(cnn_output_list[0].shape)
            )

        return torch.cat(cnn_output_list, dim=channels_dim)


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
        """An RNN for encoding the state in RL. Supports masking the hidden
        state during various timesteps in the forward lass.

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
        """Mask input hidden states given `masks`. Useful when masks represent
        steps on which a task has completed.

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
                    v * masks  # type:ignore
                    + (1.0 - masks) * (self.init_hidden_state.repeat(1, v.shape[1], 1))  # type: ignore
                    for v in hidden_states  # type:ignore
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
        # x is a (T, N, -1) tensor NO LONGER flattened to (T * N, -1)
        n = hidden_states.size(1)
        t = int(x.size(0) / n)
        assert (t, n) == x.shape[
            :2
        ], "expected shape {} not matching with input {}".format((t, n), x.shape[:2])

        # # unflatten
        # x = cast(torch.FloatTensor, x.view(t, n, x.size(1)))
        # masks = cast(torch.FloatTensor, masks.view(t, n))

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
        # x = cast(torch.FloatTensor, torch.cat(outputs, dim=0).view(t * n, -1))
        x = cast(torch.FloatTensor, torch.cat(outputs, dim=0))
        hidden_states = self._pack_hidden(unpacked_hidden_states)
        return x, hidden_states

    def forward(  # type: ignore
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


class LinearActorCritic(ActorCriticModel[CategoricalDistr]):
    def __init__(
        self,
        input_key: str,
        action_space: gym.spaces.Discrete,
        observation_space: SpaceDict,
    ):
        super().__init__(action_space=action_space, observation_space=observation_space)

        assert (
            input_key in observation_space.spaces
        ), "LinearActorCritic expects only a single observational input."
        self.key = input_key

        box_space: gym.spaces.Box = observation_space[self.key]
        assert isinstance(box_space, gym.spaces.Box), (
            "LinearActorCritic requires that"
            "observation space corresponding to the input key is a Box space."
        )
        assert len(box_space.shape) == 1
        self.in_dim = box_space.shape[0]

        self.linear = nn.Linear(self.in_dim, action_space.n + 1)

        nn.init.orthogonal_(self.linear.weight)
        nn.init.constant_(self.linear.bias, 0)

    @property
    def recurrent_hidden_state_size(self) -> int:
        return 0

    def forward(  # type: ignore
        self,
        observations: Dict[str, torch.FloatTensor],
        recurrent_hidden_states: torch.FloatTensor,
        prev_actions: torch.LongTensor,
        masks: torch.FloatTensor,
        **kwargs
    ) -> Tuple[ActorCriticOutput[DistributionType], Any]:
        out = self.linear(observations[self.key])

        # noinspection PyArgumentList
        return (
            ActorCriticOutput(
                distributions=CategoricalDistr(logits=out[:, :-1]),
                values=cast(torch.FloatTensor, out[:, -1:]),
                extras={},
            ),
            None,
        )


class RNNActorCritic(ActorCriticModel[CategoricalDistr]):
    def __init__(
        self,
        input_key: str,
        action_space: gym.spaces.Discrete,
        observation_space: SpaceDict,
        hidden_size: int = 128,
        num_layers: int = 1,
        rnn_type: str = "GRU",
        head_type: Callable[
            ..., ActorCriticModel[CategoricalDistr]
        ] = LinearActorCritic,
    ):
        super().__init__(action_space=action_space, observation_space=observation_space)
        self.hidden_size = hidden_size
        self.rnn_type = rnn_type

        assert (
            input_key in observation_space.spaces
        ), "LinearActorCritic expects only a single observational input."
        self.key = input_key

        box_space: gym.spaces.Box = observation_space[self.key]
        assert isinstance(box_space, gym.spaces.Box), (
            "RNNActorCritic requires that"
            "observation space corresponding to the input key is a Box space."
        )
        assert len(box_space.shape) == 1
        self.in_dim = box_space.shape[0]

        self.state_encoder = RNNStateEncoder(
            input_size=self.in_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            rnn_type=rnn_type,
            trainable_masked_hidden_state=True,
        )

        self.head_key = "{}_{}".format("rnn", input_key)
        self._head_observation_dict: Dict[str, Optional[torch.Tensor]] = {
            self.head_key: None
        }

        self.ac_nonrecurrent_head: ActorCriticModel[CategoricalDistr] = head_type(
            input_key=self.head_key,
            action_space=action_space,
            observation_space=SpaceDict(
                {
                    self.head_key: gym.spaces.Box(
                        low=np.float32(0.0), high=np.float32(1.0), shape=(hidden_size,)
                    )
                }
            ),
        )

    @property
    def recurrent_hidden_state_size(self) -> int:
        return self.hidden_size

    @property
    def num_recurrent_layers(self) -> int:
        return self.state_encoder.num_recurrent_layers

    def forward(  # type: ignore
        self,
        observations: Dict[str, torch.FloatTensor],
        recurrent_hidden_states: torch.FloatTensor,
        prev_actions: torch.LongTensor,
        masks: torch.FloatTensor,
        **kwargs
    ) -> Tuple[ActorCriticOutput[DistributionType], Any]:

        rnn_out, recurrent_hidden_states = self.state_encoder(
            x=observations[self.key],
            hidden_states=recurrent_hidden_states,
            masks=masks,
        )

        self._head_observation_dict[self.head_key] = rnn_out

        out, _ = self.ac_nonrecurrent_head(
            observations=self._head_observation_dict,
            recurrent_hidden_states=None,
            prev_actions=prev_actions,
            masks=masks,
        )

        self._head_observation_dict[self.head_key] = None

        # noinspection PyArgumentList
        return (
            out,
            recurrent_hidden_states,
        )
