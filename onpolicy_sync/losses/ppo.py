from typing import Dict, Union

import torch
import typing

from onpolicy_sync.losses.abstract_loss import AbstractActorCriticLoss
from rl_base.common import ActorCriticOutput
from rl_base.distributions import CategoricalDistr


class PPO(AbstractActorCriticLoss):
    """Implementation of the Proximal Policy Optimization loss."""

    def __init__(
        self,
        clip_param,
        value_loss_coef,
        entropy_coef,
        use_clipped_value_loss=True,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.clip_param = clip_param
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.use_clipped_value_loss = use_clipped_value_loss

    def loss(  # type: ignore
        self,
        batch: Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]],
        actor_critic_output: ActorCriticOutput[CategoricalDistr],
        *args,
        **kwargs
    ):
        actions = typing.cast(torch.LongTensor, batch["actions"])
        values = actor_critic_output.values
        dist_entropy: torch.FloatTensor = actor_critic_output.distributions.entropy().mean()
        action_log_probs = actor_critic_output.distributions.log_probs(actions)

        ratio = torch.exp(action_log_probs - batch["old_action_log_probs"])
        surr1 = ratio * batch["norm_adv_targ"]
        surr2 = (
            torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
            * batch["norm_adv_targ"]
        )
        action_loss = -torch.min(surr1, surr2).mean()

        if self.use_clipped_value_loss:
            value_pred_clipped = batch["values"] + (values - batch["values"]).clamp(
                -self.clip_param, self.clip_param
            )
            value_losses = (values - batch["returns"]).pow(2)
            value_losses_clipped = (value_pred_clipped - batch["returns"]).pow(2)
            value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()
        else:
            value_loss = (
                0.5
                * (typing.cast(torch.FloatTensor, batch["returns"]) - values)
                .pow(2)
                .mean()
            )

        total_loss = (
            value_loss * self.value_loss_coef
            + action_loss
            - dist_entropy * self.entropy_coef
        )

        return (
            total_loss,
            {
                "ppo_total": total_loss.item(),
                "value": value_loss.item(),
                "action": action_loss.item(),
                "entropy": -dist_entropy.item(),
            },
        )


PPOConfig = dict(clip_param=0.1, value_loss_coef=0.5, entropy_coef=0.01)
