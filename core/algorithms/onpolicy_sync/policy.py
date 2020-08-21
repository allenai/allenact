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

from core.base_abstractions.misc import ActorCriticOutput, Memory
from core.base_abstractions.distributions import CategoricalDistr

DistributionType = TypeVar("DistributionType")

MemoryDimType = Tuple[str, Optional[int]]
MemoryShapeType = Tuple[MemoryDimType, ...]
MemorySpecType = Tuple[MemoryShapeType, torch.dtype]
FullMemorySpecType = Dict[str, MemorySpecType]

ObservationType = Dict[str, Union[torch.FloatTensor, Dict[str, Any]]]


class ActorCriticModel(Generic[DistributionType], nn.Module):
    """Abstract class defining a deep (recurrent) actor critic agent.

    When defining a new agent, you should over subclass this class and implement the abstract methods.

    # Attributes

    action_space : The space of actions available to the agent. Currently only discrete
        actions are allowed (so this space will always be of type `gym.spaces.Discrete`).
    observation_space: The observation space expected by the agent. This is of type `gym.spaces.dict`.
    """

    def __init__(self, action_space: gym.spaces.Discrete, observation_space: SpaceDict):
        """Initializer.

        # Parameters

        action_space : The space of actions available to the agent.
        observation_space: The observation space expected by the agent.
        """
        super().__init__()
        self.action_space = action_space
        self.dim_actions = action_space.n
        self.observation_space = observation_space

    @property
    def recurrent_memory_specification(self) -> Optional[FullMemorySpecType]:
        """The memory specification for the `ActorCriticModel` including the `step` dimension.
        See docs for `_recurrent_memory_shape`

        # Returns

        The memory specification from `_recurrent_memory_shape` prepended with a `step` dimension.
        """
        spec = self._recurrent_memory_specification()

        if spec is None:
            return spec

        for key in spec:
            dims, dtype = spec[key]
            dim_to_pos = {dim[0]: it + 1 for it, dim in enumerate(dims)}

            assert (
                "step" not in dim_to_pos
            ), "`step` is automatically added and cannot be reused"

            spec[key] = ((("step", None),) + dims, dtype)
            dim_to_pos["step"] = 0

            assert (
                "sampler" in dim_to_pos
            ), "`sampler` dim must be defined (right before `agent` if present)"

            assert (
                "agent" not in dim_to_pos
                or dim_to_pos["agent"] == dim_to_pos["sampler"] + 1
            ), "`agent` dim must be right after `sampler`"

        return spec

    @abc.abstractmethod
    def _recurrent_memory_specification(self) -> Optional[FullMemorySpecType]:
        """Implementation of memory specification for the `ActorCriticModel`.

        # Returns

        If None, it indicates the model is memory-less.
        Otherwise, it is a one-level dictionary with string keys (memory type identification) and
        tuple values (memory type specification). Each specification tuple contains:
        1. Memory type named shape, e.g.
        `(("layer", 1), ("sampler", None), ("agent", 2), ("hidden", 32))`
        for a two-agent GRU memory, where
        the `sampler` dimension placeholder *always* precedes the optional `agent` dimension;
        the optional `agent` dimension has the number of agents in the model and is *always* the one after `sampler`;
        and `layer` and `hidden` correspond to the standard RNN hidden state parametrization.
        2. The data type, e.g. `torch.float32`.

        The inclusion of the `agent` dimension implies we are also including that dimension in our observation tensors.
        For a single-agent ActorCritic model it is often more convenient to skip the agent dimension, e.g.
        `(("layer", 1), ("sampler", None), ("hidden", 32))` for a GRU memory.
        """
        raise NotImplementedError()

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

        args : extra args.
        kwargs : extra kwargs.

        # Returns

        A tuple whose first element is an object of class ActorCriticOutput which stores
        the agent's probability distribution over possible actions, the agent's value for the
        state, and any extra information needed for loss computations. The second element
        may be any representation of the agent's hidden states.
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

        if len(out.shape) == 3:
            out = out.unsqueeze(-2)  # Enforce agent dimension

        logits = out[..., :-1]
        values = out[..., -1:]
        # noinspection PyArgumentList
        return CategoricalDistr(logits=logits), values


class LinearCriticHead(nn.Module):
    def __init__(self, input_size: int):
        super().__init__()
        self.fc = nn.Linear(input_size, 1)
        nn.init.orthogonal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
        out = self.fc(x)

        if len(out.shape) == 3:
            out = out.unsqueeze(-2)  # Enforce agent dimension

        return out


class LinearActorHead(nn.Module):
    def __init__(self, num_inputs: int, num_outputs: int):
        super().__init__()

        self.linear = nn.Linear(num_inputs, num_outputs)
        nn.init.orthogonal_(self.linear.weight, gain=0.01)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, x: torch.FloatTensor):  # type: ignore
        x = self.linear(x)  # type:ignore

        if len(x.shape) == 3:
            x = cast(torch.FloatTensor, x.unsqueeze(-2))  # Enforce agent dimension

        # noinspection PyArgumentList
        return CategoricalDistr(logits=x)
