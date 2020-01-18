from typing import Dict, Union

import torch

from onpolicy_sync.losses.abstract_loss import AbstractActorCriticLoss
from rl_base.common import ActorCriticOutput
from rl_base.distributions import CategoricalDistr


class Imitation(AbstractActorCriticLoss):
    def loss(
        self,
        batch: Dict[str, Union[Dict[str, torch.Tensor], torch.Tensor]],
        actor_critic_output: ActorCriticOutput[CategoricalDistr],
        *args,
        **kwargs
    ):
        if "expert_action" in batch["observations"]:
            expert_actions_and_mask = batch["observations"]["expert_action"]
            assert expert_actions_and_mask.shape[-1] == 2
            expert_actions_and_mask_reshaped = expert_actions_and_mask.view(-1, 2)

            expert_actions = expert_actions_and_mask_reshaped[:, 0].view(
                *expert_actions_and_mask.shape[:-1], 1
            )
            expert_actions_masks = (
                expert_actions_and_mask_reshaped[:, 1]
                .float()
                .view(*expert_actions_and_mask.shape[:-1], 1)
            )

            expert_successes = expert_actions_masks.sum()
            if expert_successes.item() == 0:
                return 0, {}

            total_loss = -(
                expert_actions_masks
                * actor_critic_output.distributions.log_probs(expert_actions)
            ).sum() / torch.clamp(expert_successes, min=1)
        elif "expert_policy" in batch["observations"]:
            raise NotImplementedError()
            # expert_policies = batch["observations"]["expert_policy"]
            # total_loss = (
            #     -(actor_critic_output.distributions.log_probs_tensor * expert_policies)
            #     .sum(-1)
            #     .mean()
            # )
        else:
            raise NotImplementedError(
                "Imitation loss requires either `expert_action` or `expert_policy`"
                " sensor to be active."
            )

        return (
            total_loss,
            {"expert_cross_entropy": total_loss.item(),},
        )
