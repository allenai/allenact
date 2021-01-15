from typing import (
    Dict,
    Union,
    Optional,
    Tuple,
    Any,
)

import torch
from torch import nn
import gym

from core.algorithms.onpolicy_sync.policy import (
    ActorCriticModel,
    DistributionType,
)
from plugins.gym_plugin.gym_distributions import GaussianDistr
from core.base_abstractions.misc import ActorCriticOutput, Memory


class MemorylessActorCritic(ActorCriticModel[GaussianDistr]):
    def __init__(
        self,
        input_uuid: str,
        action_space: gym.spaces.Box,
        observation_space: gym.spaces.Dict,
        action_std: float = 0.5,
    ):
        super().__init__(action_space, observation_space)

        self.input_uuid = input_uuid

        assert len(observation_space[self.input_uuid].shape) == 1
        state_dim = observation_space[self.input_uuid].shape[0]
        assert len(action_space.shape) == 1
        action_dim = action_space.shape[0]

        # action mean range -1 to 1
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, action_dim),
            nn.Tanh(),
        )

        # critic
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
        )

        # gates
        self.gates = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 2),
            nn.Sigmoid(),
        )

        # maximum standard deviation
        self.register_buffer(
            "action_std",
            torch.tensor([action_std] * action_dim).view(1, 1, -1),
            persistent=False,
        )

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
        gates = self.gates(observations[self.input_uuid])

        return (
            ActorCriticOutput(
                GaussianDistr(
                    loc=means * gates[..., :1], scale=self.action_std * gates[..., 1:]
                ),
                values,
                {},
            ),
            None,  # no Memory
        )
