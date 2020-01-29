#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import abc
import typing

import gym
import torch
from torch import nn as nn

from rl_base.common import DistributionType, ActorCriticOutput
from rl_base.distributions import CategoricalDistr
from gym.spaces.dict import Dict as SpaceDict


class ActorCriticModel(nn.Module, typing.Generic[DistributionType]):
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
    def recurrent_hidden_state_size(self) -> int:
        """Non-negative integer corresponding to the dimension of the hidden
        state used by the agent.

        # Returns

        The hidden state dimension (non-negative integer).
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def forward(self, *args, **kwargs) -> ActorCriticOutput:
        """Transforms input observations (& previous hidden state) into action
        probabilities and the state value.

        # Parameters

        args : extra args.
        kwargs : extra kwargs.

        # Returns

        An object of class ActorCriticOutput which stores the agent's probability distribution over possible actions,
        the agent's value for the state, and any extra information needed for loss computations.
        """
        raise NotImplementedError()


class LinearCriticHead(nn.Module):
    def __init__(self, input_size: int):
        super().__init__()
        self.fc = nn.Linear(input_size, 1)
        nn.init.orthogonal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
        return self.fc(x)


class LinearActorHead(nn.Module):
    def __init__(self, num_inputs: int, num_outputs: int):
        super().__init__()

        self.linear = nn.Linear(num_inputs, num_outputs)
        nn.init.orthogonal_(self.linear.weight, gain=0.01)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, x: torch.FloatTensor):
        x = self.linear(x)
        # noinspection PyArgumentList
        return CategoricalDistr(logits=x)
