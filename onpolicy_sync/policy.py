#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import typing

import gym
from torch import nn as nn

from rl_base.common import DistributionType, ActorCriticOutput
from rl_base.distributions import CategoricalDistr
from gym.spaces.dict import Dict as SpaceDict


class ActorCriticModel(nn.Module, typing.Generic[DistributionType]):
    def __init__(self, action_space: gym.spaces.Discrete, observation_space: SpaceDict):
        super().__init__()
        self.action_space = action_space
        self.dim_actions = action_space.n
        self.observation_space = observation_space

    def recurrent_hidden_state_size(self):
        raise NotImplementedError

    def forward(self, *args, **kwargs) -> ActorCriticOutput:
        raise NotImplementedError


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

    def forward(self, x):
        x = self.linear(x)
        return CategoricalDistr(logits=x)
