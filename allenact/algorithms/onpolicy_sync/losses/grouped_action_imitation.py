import functools
from typing import Dict, cast, Sequence, Set

import torch

from allenact.algorithms.onpolicy_sync.losses.abstract_loss import (
    AbstractActorCriticLoss,
)
from allenact.algorithms.onpolicy_sync.policy import ObservationType
from allenact.base_abstractions.distributions import CategoricalDistr
from allenact.base_abstractions.misc import ActorCriticOutput


class GroupedActionImitation(AbstractActorCriticLoss):
    def __init__(
        self, nactions: int, action_groups: Sequence[Set[int]], *args, **kwargs
    ):
        super().__init__(*args, **kwargs)

        assert (
            sum(len(ag) for ag in action_groups) == nactions
            and len(functools.reduce(lambda x, y: x | y, action_groups)) == nactions
        ), f"`action_groups` (==`{action_groups}`) must be a partition of `[0, 1, 2, ..., nactions - 1]`"

        self.nactions = nactions
        self.action_groups_mask = torch.FloatTensor(
            [
                [i in action_group for i in range(nactions)]
                for action_group in action_groups
            ]
            + [[1] * nactions]
        )

    def loss(  # type: ignore
        self,
        step_count: int,
        batch: ObservationType,
        actor_critic_output: ActorCriticOutput[CategoricalDistr],
        *args,
        **kwargs,
    ):
        observations = cast(Dict[str, torch.Tensor], batch["observations"])

        assert "expert_group_action" in observations

        expert_group_actions = observations["expert_group_action"]

        # expert_group_actions = expert_group_actions + (expert_group_actions == -1).long() * (
        #     1 + self.action_groups_mask.shape[0]
        # )

        if self.action_groups_mask.get_device() != expert_group_actions.get_device():
            self.action_groups_mask = self.action_groups_mask.cuda(
                expert_group_actions.get_device()
            )

        expert_group_actions_reshaped = expert_group_actions.view(-1, 1)

        expert_group_actions_mask = self.action_groups_mask[
            expert_group_actions_reshaped
        ]

        probs_tensor = actor_critic_output.distributions.probs_tensor
        expert_group_actions_mask = expert_group_actions_mask.view(probs_tensor.shape)

        total_loss = -(
            torch.log((probs_tensor * expert_group_actions_mask).sum(-1))
        ).mean()

        return total_loss, {"grouped_action_cross_entropy": total_loss.item(),}
