"""Defining the PPO loss for actor critic type models."""

from typing import Dict, Union

import torch
import typing

from onpolicy_sync.losses.abstract_loss import AbstractActorCriticLoss
from rl_base.common import ActorCriticOutput
from rl_base.distributions import CategoricalDistr
from typing import Optional, Callable


class PPO(AbstractActorCriticLoss):
    """Implementation of the Proximal Policy Optimization loss.

    # Attributes

    clip_param : The clipping parameter to use.
    value_loss_coef : Weight of the value loss.
    entropy_coef : Weight of the entropy (encouraging) loss.
    use_clipped_value_loss : Whether or not to also clip the value loss.
    """

    def __init__(
        self,
        clip_param: float,
        value_loss_coef: float,
        entropy_coef: float,
        use_clipped_value_loss=True,
        clip_decay: Optional[Callable[[int], float]] = None,
        *args,
        **kwargs
    ):
        """Initializer.

        See the class documentation for parameter definitions.
        """
        super().__init__(*args, **kwargs)
        self.clip_param = clip_param
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.use_clipped_value_loss = use_clipped_value_loss
        self.clip_decay = clip_decay if clip_decay is not None else (lambda x: 1.0)

    def loss(  # type: ignore
        self,
        step_count: int,
        batch: Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]],
        actor_critic_output: ActorCriticOutput[CategoricalDistr],
        *args,
        **kwargs
    ):
        actions = typing.cast(torch.LongTensor, batch["actions"])
        values = actor_critic_output.values
        dist_entropy: torch.FloatTensor = actor_critic_output.distributions.entropy().mean()
        action_log_probs = actor_critic_output.distributions.log_probs(actions)

        clip_param = self.clip_param * self.clip_decay(step_count)

        ratio = torch.exp(action_log_probs - batch["old_action_log_probs"])
        surr1 = ratio * batch["norm_adv_targ"]
        surr2 = (
            torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param)
            * batch["norm_adv_targ"]
        )
        action_loss = -torch.min(surr1, surr2).mean()

        if self.use_clipped_value_loss:
            value_pred_clipped = batch["values"] + (values - batch["values"]).clamp(
                -clip_param, clip_param
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


class PPOValue(AbstractActorCriticLoss):
    """Implementation of the Proximal Policy Optimization loss.

    # Attributes

    clip_param : The clipping parameter to use.
    use_clipped_value_loss : Whether or not to also clip the value loss.
    """

    def __init__(
        self,
        clip_param: float,
        use_clipped_value_loss=True,
        clip_decay: Optional[Callable[[int], float]] = None,
        *args,
        **kwargs
    ):
        """Initializer.

        See the class documentation for parameter definitions.
        """
        super().__init__(*args, **kwargs)
        self.clip_param = clip_param
        self.use_clipped_value_loss = use_clipped_value_loss
        self.clip_decay = clip_decay if clip_decay is not None else (lambda x: 1.0)

    def loss(  # type: ignore
        self,
        step_count: int,
        batch: Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]],
        actor_critic_output: ActorCriticOutput[CategoricalDistr],
        *args,
        **kwargs
    ):
        values = actor_critic_output.values
        clip_param = self.clip_param * self.clip_decay(step_count)

        if self.use_clipped_value_loss:
            value_pred_clipped = batch["values"] + (values - batch["values"]).clamp(
                -clip_param, clip_param
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

        return (
            value_loss,
            {"value": value_loss.item(),},
        )


PPOConfig = dict(clip_param=0.1, value_loss_coef=0.5, entropy_coef=0.01)
