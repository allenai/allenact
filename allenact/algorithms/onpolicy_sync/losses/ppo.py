"""Defining the PPO loss for actor critic type models."""

from typing import Dict, Optional, Callable, cast, Tuple

import torch

from allenact.algorithms.onpolicy_sync.losses.abstract_loss import (
    AbstractActorCriticLoss,
    ObservationType,
)
from allenact.base_abstractions.distributions import CategoricalDistr
from allenact.base_abstractions.misc import ActorCriticOutput


class PPO(AbstractActorCriticLoss):
    """Implementation of the Proximal Policy Optimization loss.

    # Attributes

    clip_param : The clipping parameter to use.
    value_loss_coef : Weight of the value loss.
    entropy_coef : Weight of the entropy (encouraging) loss.
    use_clipped_value_loss : Whether or not to also clip the value loss.
    clip_decay : Callable for clip param decay factor (function of the current number of steps)
    entropy_method_name : Name of Distr's entropy method name. Default is `entropy`,
                          but we might use `conditional_entropy` for `SequentialDistr`
    """

    def __init__(
        self,
        clip_param: float,
        value_loss_coef: float,
        entropy_coef: float,
        use_clipped_value_loss=True,
        clip_decay: Optional[Callable[[int], float]] = None,
        entropy_method_name: str = "entropy",
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
        self.entropy_method_name = entropy_method_name

    def loss_per_step(
        self,
        step_count: int,
        batch: ObservationType,
        actor_critic_output: ActorCriticOutput[CategoricalDistr],
    ) -> Dict[str, Tuple[torch.Tensor, Optional[float]]]:

        actions = cast(torch.LongTensor, batch["actions"])
        values = actor_critic_output.values

        action_log_probs = actor_critic_output.distributions.log_prob(actions)
        dist_entropy: torch.FloatTensor = getattr(
            actor_critic_output.distributions, self.entropy_method_name
        )()

        def add_trailing_dims(t: torch.Tensor):
            assert len(t.shape) <= len(batch["norm_adv_targ"].shape)
            return t.view(
                t.shape + ((1,) * (len(batch["norm_adv_targ"].shape) - len(t.shape)))
            )

        dist_entropy = add_trailing_dims(dist_entropy)

        clip_param = self.clip_param * self.clip_decay(step_count)

        ratio = torch.exp(action_log_probs - batch["old_action_log_probs"])
        ratio = add_trailing_dims(ratio)

        surr1 = ratio * batch["norm_adv_targ"]
        surr2 = (
            torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param)
            * batch["norm_adv_targ"]
        )
        action_loss = -torch.min(surr1, surr2)

        if self.use_clipped_value_loss:
            value_pred_clipped = batch["values"] + (values - batch["values"]).clamp(
                -clip_param, clip_param
            )
            value_losses = (values - batch["returns"]).pow(2)
            value_losses_clipped = (value_pred_clipped - batch["returns"]).pow(2)
            value_loss = 0.5 * torch.max(value_losses, value_losses_clipped)
        else:
            value_loss = 0.5 * (cast(torch.FloatTensor, batch["returns"]) - values).pow(
                2
            )

        # noinspection PyUnresolvedReferences
        return {
            "value": (value_loss, self.value_loss_coef),
            "action": (action_loss, None),
            "entropy": (dist_entropy.mul_(-1.0), self.entropy_coef),  # type: ignore
        }

    def loss(  # type: ignore
        self,
        step_count: int,
        batch: ObservationType,
        actor_critic_output: ActorCriticOutput[CategoricalDistr],
        *args,
        **kwargs
    ):
        losses_per_step = self.loss_per_step(
            step_count=step_count, batch=batch, actor_critic_output=actor_critic_output,
        )
        losses = {
            key: (loss.mean(), weight)
            for (key, (loss, weight)) in losses_per_step.items()
        }

        total_loss = sum(
            loss * weight if weight is not None else loss
            for loss, weight in losses.values()
        )

        return (
            total_loss,
            {
                "ppo_total": cast(torch.Tensor, total_loss).item(),
                **{key: loss.item() for key, (loss, _) in losses.items()},
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
        batch: ObservationType,
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
                0.5 * (cast(torch.FloatTensor, batch["returns"]) - values).pow(2).mean()
            )

        return (
            value_loss,
            {"value": value_loss.item(),},
        )


PPOConfig = dict(clip_param=0.2, value_loss_coef=0.5, entropy_coef=0.01)
