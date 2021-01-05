#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import abc
from typing import TypeVar, Generic, Tuple, Optional, Union, Dict, cast, Any

import gym
import torch
from gym.spaces.dict import Dict as SpaceDict
from torch import nn

from core.base_abstractions.distributions import CategoricalDistr
from core.base_abstractions.misc import ActorCriticOutput, Memory
from utils.spaces_utils import unflatten as spaces_unflatten

DistributionType = TypeVar("DistributionType")

MemoryDimType = Tuple[str, Optional[int]]
MemoryShapeType = Tuple[MemoryDimType, ...]
MemorySpecType = Tuple[MemoryShapeType, torch.dtype]
FullMemorySpecType = Dict[str, MemorySpecType]

ObservationType = Dict[str, Union[torch.Tensor, Dict[str, Any]]]
ActionType = torch.Tensor


class ActorCriticModel(Generic[DistributionType], nn.Module):
    """Abstract class defining a deep (recurrent) actor critic agent.

    When defining a new agent, you should subclass this class and implement the abstract methods.

    # Attributes

    action_space : The space of actions available to the agent. This is of type `gym.spaces.Space`.
    observation_space: The observation space expected by the agent. This is of type `gym.spaces.dict`.
    """

    def __init__(self, action_space: gym.Space, observation_space: SpaceDict):
        """Initializer.

        # Parameters

        action_space : The space of actions available to the agent.
        observation_space: The observation space expected by the agent.
        """
        super().__init__()
        self.action_space = action_space
        self.observation_space = observation_space

    @property
    def recurrent_memory_specification(self) -> Optional[FullMemorySpecType]:
        """The memory specification for the `ActorCriticModel`. See docs for
        `_recurrent_memory_shape`

        # Returns

        The memory specification from `_recurrent_memory_shape`.
        """
        spec = self._recurrent_memory_specification()

        if spec is None:
            return spec

        for key in spec:
            dims, _ = spec[key]
            dim_names = [d[0] for d in dims]

            assert (
                "step" not in dim_names
            ), "`step` is automatically added and cannot be reused"

            assert "sampler" in dim_names, "`sampler` dim must be defined"

        return spec

    @abc.abstractmethod
    def _recurrent_memory_specification(self) -> Optional[FullMemorySpecType]:
        """Implementation of memory specification for the `ActorCriticModel`.

        # Returns

        If None, it indicates the model is memory-less.
        Otherwise, it is a one-level dictionary (a map) with string keys (memory type identification) and
        tuple values (memory type specification). Each specification tuple contains:
        1. Memory type named shape, e.g.
        `(("layer", 1), ("sampler", None), ("agent", 2), ("hidden", 32))`
        for a two-agent GRU memory, where
        the `sampler` dimension placeholder *always* precedes the optional `agent` dimension;
        the optional `agent` dimension has the number of agents in the model and is *always* the one after
        `sampler` if present;
        and `layer` and `hidden` correspond to the standard RNN hidden state parametrization.
        2. The data type, e.g. `torch.float32`.

        The `sampler` dimension placeholder is mandatory for all memories.

        For a single-agent ActorCritic model it is often more convenient to skip the agent dimension, e.g.
        `(("layer", 1), ("sampler", None), ("hidden", 32))` for a GRU memory.
        """
        raise NotImplementedError()

    def unflatten_actions(
        self, flattened_actions: torch.Tensor
    ) -> Union[Tuple, Dict, torch.Tensor, int, float]:
        """Unflattens a flattened actions tensor, e.g. the prev_actions tensor passed to forward.

        # Returns

        The unflattened tensors or scalars corresponding to the input flattened actions tensor.
        """
        return spaces_unflatten(self.action_space, flattened_actions)

    @abc.abstractmethod
    def forward(  # type:ignore
        self,
        observations: ObservationType,
        memory: Memory,
        prev_actions: torch.Tensor,
        masks: torch.FloatTensor,
    ) -> Tuple[ActorCriticOutput[DistributionType], Optional[Memory]]:
        """Transforms input observations (& previous hidden state) into action
        probabilities and the state value.

        # Parameters

        observations : Multi-level map from key strings to tensors of shape [steps, samplers, (agents,) ...] with the
                       current observations.
        memory : `Memory` object with recurrent memory. The shape of each tensor is determined by the corresponding
                 entry in `_recurrent_memory_specification`.
        prev_actions : tensor of shape [steps, samplers, ...] with the flattened previous actions.
        masks : tensor of shape [steps, samplers, agents, 1] with zeros indicating steps where a new episode/task
                starts.

        # Returns

        A tuple whose first element is an object of class ActorCriticOutput which stores
        the agents' probability distribution over possible actions (shape [steps, samplers, ...]),
        the agents' value for the state (shape [steps, samplers, ..., 1]), and any extra information needed for
        loss computations. The second element is an optional `Memory`, which is only used in models with recurrent
        memory.
        """
        raise NotImplementedError()


class LinearActorCriticHead(nn.Module):
    def __init__(self, input_size: int, num_actions: int):
        super().__init__()
        self.input_size = input_size
        self.num_actions = num_actions
        self.actor_and_critic = nn.Linear(input_size, 1 + num_actions)

        nn.init.orthogonal_(self.actor_and_critic.weight)
        nn.init.constant_(self.actor_and_critic.bias, 0)

    def forward(self, x):
        out = self.actor_and_critic(x)

        logits = out[..., :-1]
        values = out[..., -1:]
        # noinspection PyArgumentList
        return (
            CategoricalDistr(logits=logits),
            values.view(*values.shape[:2], -1),
        )  # steps, samplers, flattened


class LinearCriticHead(nn.Module):
    def __init__(self, input_size: int):
        super().__init__()
        self.fc = nn.Linear(input_size, 1)
        nn.init.orthogonal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
        return self.fc(x).view(*x.shape[:2], -1)  # steps, samplers, flattened


class LinearActorHead(nn.Module):
    def __init__(self, num_inputs: int, num_outputs: int):
        super().__init__()

        self.linear = nn.Linear(num_inputs, num_outputs)
        nn.init.orthogonal_(self.linear.weight, gain=0.01)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, x: torch.FloatTensor):  # type: ignore
        x = self.linear(x)  # type:ignore

        # noinspection PyArgumentList
        return CategoricalDistr(logits=x)
