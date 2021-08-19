"""
Note: I add this file just for the format consistence with other baselines in the project, so it is just the same as
`allenact_plugins.gym_models.py` so far. However, if it is in the Gym Robotics, some modification is need.
For example, for `state_dim`:
        if input_uuid == 'gym_robotics_data':
            # consider that the observation space is Dict for robotics env
            state_dim = observation_space[self.input_uuid]['observation'].shape[0]
        else:
            assert len(observation_space[self.input_uuid].shape) == 1
            state_dim = observation_space[self.input_uuid].shape[0]
"""
from typing import Dict, Union, Optional, Tuple, Any, Sequence, cast

import gym
import torch
import torch.nn as nn

from allenact.algorithms.onpolicy_sync.policy import (
    ActorCriticModel,
    DistributionType,
)
from allenact.base_abstractions.misc import ActorCriticOutput, Memory
from allenact_plugins.gym_plugin.gym_distributions import GaussianDistr


class MemorylessActorCritic(ActorCriticModel[GaussianDistr]):
    """ActorCriticModel for gym tasks with continuous control in the range [-1,
    1]."""

    def __init__(
        self,
        input_uuid: str,
        action_space: gym.spaces.Box,
        observation_space: gym.spaces.Dict,
        action_std: float = 0.5,
        mlp_hidden_dims: Sequence[int] = (64, 32),
    ):
        super().__init__(action_space, observation_space)

        self.input_uuid = input_uuid

        assert len(observation_space[self.input_uuid].shape) == 1
        state_dim = observation_space[self.input_uuid].shape[0]
        assert len(action_space.shape) == 1
        action_dim = action_space.shape[0]

        mlp_hidden_dims = (state_dim,) + tuple(mlp_hidden_dims)

        # action mean range -1 to 1
        self.actor = nn.Sequential(
            *self.make_mlp_hidden(nn.Tanh, *mlp_hidden_dims),
            nn.Linear(32, action_dim),
            nn.Tanh(),
        )

        # critic
        self.critic = nn.Sequential(
            *self.make_mlp_hidden(nn.Tanh, *mlp_hidden_dims),
            nn.Linear(32, 1),
        )

        # maximum standard deviation
        self.register_buffer(
            "action_std",
            torch.tensor([action_std] * action_dim).view(1, 1, -1),
            persistent=False,
        )

    @staticmethod
    def make_mlp_hidden(nl, *dims):
        res = []
        for it, dim in enumerate(dims[:-1]):
            res.append(
                nn.Linear(dim, dims[it + 1]),
            )
            res.append(nl())
        return res

    def _recurrent_memory_specification(self):
        return None

    def forward(  # type:ignore
        self,
        observations: Dict[str, Union[torch.FloatTensor, Dict[str, Any]]],
        memory: Memory,
        prev_actions: Any,
        masks: torch.FloatTensor,
    ) -> Tuple[ActorCriticOutput[DistributionType], Optional[Memory]]:
        means = self.actor(observations[self.input_uuid])
        values = self.critic(observations[self.input_uuid])

        return (
            ActorCriticOutput(
                cast(DistributionType, GaussianDistr(loc=means, scale=self.action_std)),
                values,
                {},
            ),
            None,  # no Memory
        )
