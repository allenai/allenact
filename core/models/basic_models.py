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
from core.base_abstractions.misc import ActorCriticOutput, Memory
from core.base_abstractions.distributions import CategoricalDistr
from utils.model_utils import make_cnn, compute_cnn_output, Flatten


class SimpleCNN(nn.Module):
    """A Simple N-Conv CNN followed by a fully connected layer. Takes in
    observations (of type gym.spaces.dict) and produces an embedding of the
    `rgb_uuid` and/or `depth_uuid` components.

    # Attributes

    observation_space : The observation_space of the agent, should have `rgb_uuid` or `depth_uuid` as
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
        flatten: bool = True,
        output_relu: bool = True,
    ):
        """Initializer.

        # Parameters

        observation_space : See class attributes documentation.
        output_size : See class attributes documentation.
        """
        super().__init__()

        self.rgb_uuid = rgb_uuid
        if self.rgb_uuid in observation_space.spaces:
            self._n_input_rgb = observation_space.spaces[self.rgb_uuid].shape[2]
            assert self._n_input_rgb >= 0
        else:
            self._n_input_rgb = 0

        self.depth_uuid = depth_uuid
        if self.depth_uuid in observation_space.spaces:
            self._n_input_depth = observation_space.spaces[self.depth_uuid].shape[2]
            assert self._n_input_depth >= 0
        else:
            self._n_input_depth = 0

        if not self.is_blind:
            # hyperparameters for layers
            self._cnn_layers_channels = list(layer_channels)
            self._cnn_layers_kernel_size = list(kernel_sizes)
            self._cnn_layers_stride = list(layers_stride)
            self._cnn_layers_paddings = list(paddings)
            self._cnn_layers_dilations = list(dilations)

            if self._n_input_rgb > 0:
                input_rgb_cnn_dims = np.array(
                    observation_space.spaces[self.rgb_uuid].shape[:2], dtype=np.float32
                )
                self.rgb_cnn = self.make_cnn_from_params(
                    output_size=output_size,
                    input_dims=input_rgb_cnn_dims,
                    input_channels=self._n_input_rgb,
                    flatten=flatten,
                    output_relu=output_relu,
                )

            if self._n_input_depth > 0:
                input_depth_cnn_dims = np.array(
                    observation_space.spaces[self.depth_uuid].shape[:2],
                    dtype=np.float32,
                )
                self.depth_cnn = self.make_cnn_from_params(
                    output_size=output_size,
                    input_dims=input_depth_cnn_dims,
                    input_channels=self._n_input_depth,
                    flatten=flatten,
                    output_relu=output_relu,
                )

    def make_cnn_from_params(
        self,
        output_size: int,
        input_dims: np.ndarray,
        input_channels: int,
        flatten: bool,
        output_relu: bool,
    ) -> nn.Module:
        output_dims = input_dims
        for kernel_size, stride, padding, dilation in zip(
            self._cnn_layers_kernel_size,
            self._cnn_layers_stride,
            self._cnn_layers_paddings,
            self._cnn_layers_dilations,
        ):
            # noinspection PyUnboundLocalVariable
            output_dims = self._conv_output_dim(
                dimension=output_dims,
                padding=np.array(padding, dtype=np.float32),
                dilation=np.array(dilation, dtype=np.float32),
                kernel_size=np.array(kernel_size, dtype=np.float32),
                stride=np.array(stride, dtype=np.float32),
            )

        # noinspection PyUnboundLocalVariable
        cnn = make_cnn(
            input_channels=input_channels,
            layer_channels=self._cnn_layers_channels,
            kernel_sizes=self._cnn_layers_kernel_size,
            strides=self._cnn_layers_stride,
            paddings=self._cnn_layers_paddings,
            dilations=self._cnn_layers_dilations,
            output_height=output_dims[0],
            output_width=output_dims[1],
            output_channels=output_size,
            flatten=flatten,
            output_relu=output_relu,
        )
        self.layer_init(cnn)

        return cnn

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
        """True if the observation space doesn't include `self.rgb_uuid` or
        `self.depth_uuid`."""
        return self._n_input_rgb + self._n_input_depth == 0

    def forward(self, observations: Dict[str, torch.Tensor]):  # type: ignore
        if self.is_blind:
            return None

        def check_use_agent(new_setting):
            if use_agent is not None:
                assert (
                    use_agent is new_setting
                ), "rgb and depth must both use an agent dim or none"
            return new_setting

        cnn_output_list: List[torch.Tensor] = []
        use_agent: Optional[bool] = None

        if self._n_input_rgb > 0:
            use_agent = check_use_agent(len(observations[self.rgb_uuid]) == 6)
            cnn_output_list.append(
                compute_cnn_output(self.rgb_cnn, observations[self.rgb_uuid])
            )

        if self._n_input_depth > 0:
            use_agent = check_use_agent(len(observations[self.depth_uuid]) == 6)
            cnn_output_list.append(
                compute_cnn_output(self.depth_cnn, observations[self.depth_uuid])
            )

        if use_agent:
            channels_dim = 3  # [step, sampler, agent, channel (, height, width)]
        else:
            channels_dim = 2  # [step, sampler, channel (, height, width)]

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
        """Stacks hidden states in an LSTM together (if using a GRU rather
        than an LSTM this is just the identity).

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
        return cast(torch.FloatTensor, hidden_states)

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
        """Forward for a single-step input."""
        (
            x,
            hidden_states,
            masks,
            mem_agent,
            obs_agent,
            nsteps,
            nsamplers,
            nagents,
        ) = self.adapt_input(x, hidden_states, masks)

        unpacked_hidden_states = self._unpack_hidden(hidden_states)

        x, unpacked_hidden_states = self.rnn(
            x,
            self._mask_hidden(
                unpacked_hidden_states, cast(torch.FloatTensor, masks[0].view(1, -1, 1))
            ),
        )

        return self.adapt_result(
            x,
            self._pack_hidden(unpacked_hidden_states),
            mem_agent,
            obs_agent,
            nsteps,
            nsamplers,
            nagents,
        )

    def adapt_input(
        self,
        x: torch.FloatTensor,
        hidden_states: torch.FloatTensor,
        masks: torch.FloatTensor,
    ) -> Tuple[
        torch.FloatTensor,
        torch.FloatTensor,
        torch.FloatTensor,
        bool,
        bool,
        int,
        int,
        int,
    ]:
        nsteps, nsamplers, nagents = masks.shape[:3]  # masks always have all dimensions

        assert len(hidden_states.shape) in [
            3,
            4,
        ], "hidden_states must be [layer, sampler, hidden] or [layer, sampler, agent, hidden]"

        assert len(x.shape) in [
            3,
            4,
        ], "observations must be [step, sampler, data] or [step, sampler, agent, data]"

        mem_agent: bool
        if len(hidden_states.shape) == 4:  # [layer, sampler, agent, hidden]
            mem_agent = True
        else:  # [layer, sampler, hidden]
            mem_agent = False

        obs_agent: bool
        if len(x.shape) == 4:  # [step, sampler, agent, dims]
            obs_agent = True
        else:  # [step, sampler, agent, dims]
            obs_agent = False

        # Flatten (nsamplers, nagents)
        x = x.view(nsteps, nsamplers * nagents, -1)  # type:ignore
        masks = masks.view(nsteps, nsamplers * nagents)  # type:ignore

        # Flatten (nsamplers, nagents) and remove step dim
        hidden_states = hidden_states.view(  # type:ignore
            self.num_recurrent_layers, nsamplers * nagents, -1
        )

        return x, hidden_states, masks, mem_agent, obs_agent, nsteps, nsamplers, nagents

    def adapt_result(
        self,
        outputs: torch.FloatTensor,
        hidden_states: torch.FloatTensor,
        mem_agent: bool,
        obs_agent: bool,
        nsteps: int,
        nsamplers: int,
        nagents: int,
    ) -> Tuple[
        torch.FloatTensor, torch.FloatTensor,
    ]:
        output_dims = (nsteps, nsamplers) + ((nagents, -1) if obs_agent else (-1,))
        hidden_dims = (self.num_recurrent_layers, nsamplers) + (
            (nagents, -1) if mem_agent else (-1,)
        )

        outputs = cast(torch.FloatTensor, outputs.view(*output_dims))
        hidden_states = cast(torch.FloatTensor, hidden_states.view(*hidden_dims),)

        return outputs, hidden_states

    def seq_forward(  # type: ignore
        self,
        x: torch.FloatTensor,
        hidden_states: torch.FloatTensor,
        masks: torch.FloatTensor,
    ) -> Tuple[
        torch.FloatTensor, Union[torch.FloatTensor, Tuple[torch.FloatTensor, ...]]
    ]:
        """Forward for a sequence of length T.

        # Parameters

        x : (Steps, Samplers, Agents, -1) tensor.
        hidden_states : The starting hidden states.
        masks : A (Steps, Samplers, Agents) tensor.
            The masks to be applied to hidden state at every timestep, equal to 0 whenever the previous step finalized
            the task, 1 elsewhere.
        """
        (
            x,
            hidden_states,
            masks,
            mem_agent,
            obs_agent,
            nsteps,
            nsamplers,
            nagents,
        ) = self.adapt_input(x, hidden_states, masks)

        # steps in sequence which have zero for any episode. Assume t=0 has
        # a zero in it.
        has_zeros = (masks[1:] == 0.0).any(dim=-1).nonzero().squeeze().cpu()
        # +1 to correct the masks[1:]
        if has_zeros.dim() == 0:
            # handle scalar
            has_zeros = [has_zeros.item() + 1]  # type: ignore
        else:
            has_zeros = (has_zeros + 1).numpy().tolist()
        # add t=0 and t=T to the list
        has_zeros = cast(List[int], [0] + has_zeros + [nsteps])

        unpacked_hidden_states = self._unpack_hidden(
            cast(torch.FloatTensor, hidden_states)
        )

        outputs = []
        for i in range(len(has_zeros) - 1):
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

        return self.adapt_result(
            cast(torch.FloatTensor, torch.cat(outputs, dim=0)),
            self._pack_hidden(unpacked_hidden_states),
            mem_agent,
            obs_agent,
            nsteps,
            nsamplers,
            nagents,
        )

    def forward(  # type: ignore
        self,
        x: torch.FloatTensor,
        hidden_states: torch.FloatTensor,
        masks: torch.FloatTensor,
    ) -> Tuple[
        torch.FloatTensor, Union[torch.FloatTensor, Tuple[torch.FloatTensor, ...]]
    ]:
        nsteps = masks.shape[0]
        if nsteps == 1:
            return self.single_forward(x, hidden_states, masks)
        return self.seq_forward(x, hidden_states, masks)


class LinearActorCritic(ActorCriticModel[CategoricalDistr]):
    def __init__(
        self,
        input_uuid: str,
        action_space: gym.spaces.Discrete,
        observation_space: SpaceDict,
    ):
        super().__init__(action_space=action_space, observation_space=observation_space)

        assert (
            input_uuid in observation_space.spaces
        ), "LinearActorCritic expects only a single observational input."
        self.input_uuid = input_uuid

        box_space: gym.spaces.Box = observation_space[self.input_uuid]
        assert isinstance(box_space, gym.spaces.Box), (
            "LinearActorCritic requires that"
            "observation space corresponding to the input uuid is a Box space."
        )
        assert len(box_space.shape) == 1
        self.in_dim = box_space.shape[0]

        self.linear = nn.Linear(self.in_dim, action_space.n + 1)

        nn.init.orthogonal_(self.linear.weight)
        nn.init.constant_(self.linear.bias, 0)

    def _recurrent_memory_specification(self):
        return None

    def forward(self, observations, memory, prev_actions, masks):
        out = self.linear(observations[self.input_uuid])

        assert len(out.shape) in [
            3,
            4,
        ], "observations must be [step, sampler, data] or [step, sampler, agent, data]"

        if len(out.shape) == 3:
            # [step, sampler, data] -> [step, sampler, agent, data]
            out = out.unsqueeze(-2)

        # noinspection PyArgumentList
        return (
            ActorCriticOutput(
                distributions=CategoricalDistr(logits=out[..., :-1]),
                values=cast(torch.FloatTensor, out[..., -1:]),
                extras={},
            ),
            None,
        )


class RNNActorCritic(ActorCriticModel[CategoricalDistr]):
    def __init__(
        self,
        input_uuid: str,
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
            input_uuid in observation_space.spaces
        ), "LinearActorCritic expects only a single observational input."
        self.input_uuid = input_uuid

        box_space: gym.spaces.Box = observation_space[self.input_uuid]
        assert isinstance(box_space, gym.spaces.Box), (
            "RNNActorCritic requires that"
            "observation space corresponding to the input uuid is a Box space."
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

        self.head_uuid = "{}_{}".format("rnn", input_uuid)

        self.ac_nonrecurrent_head: ActorCriticModel[CategoricalDistr] = head_type(
            input_uuid=self.head_uuid,
            action_space=action_space,
            observation_space=SpaceDict(
                {
                    self.head_uuid: gym.spaces.Box(
                        low=np.float32(0.0), high=np.float32(1.0), shape=(hidden_size,)
                    )
                }
            ),
        )

        self.memory_key = "rnn"

    @property
    def recurrent_hidden_state_size(self) -> int:
        return self.hidden_size

    @property
    def num_recurrent_layers(self) -> int:
        return self.state_encoder.num_recurrent_layers

    def _recurrent_memory_specification(self):
        return {
            self.memory_key: (
                (
                    ("layer", self.num_recurrent_layers),
                    ("sampler", None),
                    ("hidden", self.recurrent_hidden_state_size),
                ),
                torch.float32,
            )
        }

    def forward(  # type:ignore
        self,
        observations: Dict[str, Union[torch.FloatTensor, Dict[str, Any]]],
        memory: Memory,
        prev_actions: torch.Tensor,
        masks: torch.FloatTensor,
    ) -> Tuple[ActorCriticOutput[DistributionType], Optional[Memory]]:
        rnn_out, mem_return = self.state_encoder(
            x=observations[self.input_uuid],
            hidden_states=memory.tensor(self.memory_key),
            masks=masks,
        )

        out, _ = self.ac_nonrecurrent_head(
            observations={self.head_uuid: rnn_out},
            memory=None,
            prev_actions=prev_actions,
            masks=masks,
        )

        # noinspection PyArgumentList
        return (
            out,
            memory.set_tensor(self.memory_key, mem_return),
        )
