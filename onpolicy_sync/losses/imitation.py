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
            expert_actions = batch["observations"]["expert_action"]
            total_loss = -(
                actor_critic_output.distributions.log_probs(expert_actions)
            ).mean()
        elif "expert_policy" in batch["observations"]:
            expert_policies = batch["observations"]["expert_policy"]
            total_loss = (
                -(actor_critic_output.distributions.log_probs_tensor * expert_policies)
                .sum(-1)
                .mean()
            )
        else:
            raise NotImplementedError(
                "Imitation loss requires either `expert_action` or `expert_policy`"
                " sensor to be active."
            )

        return (
            total_loss,
            {"expert_cross_entropy": total_loss.item(),},
        )
