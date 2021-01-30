from typing import Optional, Tuple, cast

import gym
import torch
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
        input_key: str,
        action_space: gym.spaces.Discrete,
        observation_space: SpaceDict,
        ensure_same_weights: bool = True,
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

        self.num_actions = action_space.n
        self.linear = torch.nn.Linear(self.in_dim, 2 * self.num_actions + 1)

        torch.nn.init.orthogonal_(self.linear.weight)
        if ensure_same_weights:
            # Ensure main actor / auxiliary actor start with the same weights
            self.linear.weight.data[self.num_actions : -1, :] = self.linear.weight[
                : self.num_actions, :
            ]
        torch.nn.init.constant_(self.linear.bias, 0)

    @property
    def recurrent_hidden_state_size(self) -> int:
        return 0

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
        out = self.linear(cast(torch.Tensor, observations[self.key]))

        assert len(out.shape) in [
            3,
            4,
        ], "observations must be [step, sampler, data] or [step, sampler, agent, data]"

        if len(out.shape) == 3:
            # [step, sampler, data] -> [step, sampler, agent, data]
            out = out.unsqueeze(-2)

        main_logits = out[..., : self.num_actions]
        aux_logits = out[..., self.num_actions : -1]
        values = out[..., -1:]

        # noinspection PyArgumentList
        return (
            ActorCriticOutput(
                distributions=CategoricalDistr(
                    logits=main_logits
                ),  # step x sampler x ...
                values=cast(
                    torch.FloatTensor, values.view(values.shape[:2] + (-1,))
                ),  # step x sampler x flattened
                extras={
                    "auxiliary_distributions": CategoricalDistr(logits=aux_logits),
                },
            ),
            None,
        )
