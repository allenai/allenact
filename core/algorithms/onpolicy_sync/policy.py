#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import abc
from typing import TypeVar, Generic, Tuple, Optional, Union, Dict, Sequence

import gym
import torch
from gym.spaces.dict import Dict as SpaceDict
from torch import nn

from core.base_abstractions.misc import ActorCriticOutput, Memory
from core.base_abstractions.distributions import CategoricalDistr

DistributionType = TypeVar("DistributionType")


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
    @abc.abstractmethod
    def recurrent_hidden_state_size(
        self,
    ) -> Union[int, Dict[str, Tuple[Sequence[Tuple[str, Optional[int]]], torch.dtype]]]:
        """Non-negative integer corresponding to the dimension of the hidden
        state used by the agent or mapping from string memory names to Tuples
        of (0) sequences of axes dimensions excluding sampler axis; (1)
        position for sampler axis; and (2) data types.

        # Returns

        The hidden state dimension (non-negative integer) or dict with memory specification.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def forward(
        self, *args, **kwargs
    ) -> Tuple[
        ActorCriticOutput[DistributionType], Optional[Union[torch.Tensor, Memory]]
    ]:
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
            x = x.unsqueeze(-2)  # Enforce agent dimension

        # noinspection PyArgumentList
        return CategoricalDistr(logits=x)
