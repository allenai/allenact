from typing import Optional, Tuple, cast

import gym
import torch
import torch.nn as nn
from gym.spaces.dict import Dict as SpaceDict

from allenact.algorithms.onpolicy_sync.policy import (
    ActorCriticModel,
    Memory,
    ObservationType,
)
from allenact.base_abstractions.distributions import CategoricalDistr
from allenact.base_abstractions.misc import ActorCriticOutput, DistributionType


class LinearAdvisorActorCritic(ActorCriticModel[CategoricalDistr]):
    def __init__(
        self,
        input_uuid: str,
        action_space: gym.spaces.Discrete,
        observation_space: SpaceDict,
        ensure_same_init_aux_weights: bool = True,
    ):
        super().__init__(action_space=action_space, observation_space=observation_space)

        assert (
            input_uuid in observation_space.spaces
        ), "LinearActorCritic expects only a single observational input."
        self.input_uuid = input_uuid

        box_space: gym.spaces.Box = observation_space[self.input_uuid]
        assert isinstance(box_space, gym.spaces.Box), (
            "LinearActorCritic requires that"
            "observation space corresponding to the input key is a Box space."
        )
        assert len(box_space.shape) == 1
        self.in_dim = box_space.shape[0]

        self.num_actions = action_space.n
        self.linear = nn.Linear(self.in_dim, 2 * self.num_actions + 1)

        nn.init.orthogonal_(self.linear.weight)
        if ensure_same_init_aux_weights:
            # Ensure main actor / auxiliary actor start with the same weights
            self.linear.weight.data[self.num_actions : -1, :] = self.linear.weight[
                : self.num_actions, :
            ]
        nn.init.constant_(self.linear.bias, 0)

    # noinspection PyMethodMayBeStatic
    def _recurrent_memory_specification(self):
        return None

    def forward(  # type:ignore
        self,
        observations: ObservationType,
        memory: Memory,
        prev_actions: torch.Tensor,
        masks: torch.FloatTensor,
    ) -> Tuple[ActorCriticOutput[DistributionType], Optional[Memory]]:
        out = self.linear(cast(torch.Tensor, observations[self.input_uuid]))

        main_logits = out[..., : self.num_actions]
        aux_logits = out[..., self.num_actions : -1]
        values = out[..., -1:]

        # noinspection PyArgumentList
        return (
            ActorCriticOutput(
                distributions=cast(
                    DistributionType, CategoricalDistr(logits=main_logits)
                ),  # step x sampler x ...
                values=cast(
                    torch.FloatTensor, values.view(values.shape[:2] + (-1,))
                ),  # step x sampler x flattened
                extras={"auxiliary_distributions": CategoricalDistr(logits=aux_logits)},
            ),
            None,
        )
