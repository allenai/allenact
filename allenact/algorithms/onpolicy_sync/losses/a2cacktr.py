"""Implementation of A2C and ACKTR losses."""
from typing import cast, Tuple, Dict, Optional

import torch

from allenact.algorithms.onpolicy_sync.losses.abstract_loss import (
    AbstractActorCriticLoss,
    ObservationType,
)
from allenact.base_abstractions.distributions import CategoricalDistr
from allenact.base_abstractions.misc import ActorCriticOutput
from allenact.utils.system import get_logger


class A2CACKTR(AbstractActorCriticLoss):
    """Class implementing A2C and ACKTR losses.

    # Attributes

    acktr : `True` if should use ACKTR loss (currently not supported), otherwise uses A2C loss.
    value_loss_coef : Weight of value loss.
    entropy_coef : Weight of entropy (encouraging) loss.
    """

    def __init__(
        self,
        value_loss_coef,
        entropy_coef,
        acktr=False,
        entropy_method_name: str = "entropy",
        *args,
        **kwargs,
    ):
        """Initializer.

        See class documentation for parameter definitions.
        """
        super().__init__(*args, **kwargs)
        self.acktr = acktr
        self.loss_key = "a2c_total" if not acktr else "aktr_total"

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.entropy_method_name = entropy_method_name

    def loss_per_step(  # type: ignore
        self,
        step_count: int,
        batch: ObservationType,
        actor_critic_output: ActorCriticOutput[CategoricalDistr],
    ) -> Dict[str, Tuple[torch.Tensor, Optional[float]]]:
        actions = cast(torch.LongTensor, batch["actions"])
        values = actor_critic_output.values
        action_log_probs = actor_critic_output.distributions.log_prob(actions)
        action_log_probs = action_log_probs.view(
            action_log_probs.shape
            + (1,)
            * (
                len(cast(torch.Tensor, batch["adv_targ"]).shape)
                - len(action_log_probs.shape)
            )
        )

        dist_entropy: torch.FloatTensor = getattr(
            actor_critic_output.distributions, self.entropy_method_name
        )()
        dist_entropy = dist_entropy.view(
            dist_entropy.shape
            + ((1,) * (len(action_log_probs.shape) - len(dist_entropy.shape)))
        )

        value_loss = 0.5 * (cast(torch.FloatTensor, batch["returns"]) - values).pow(2)

        # TODO: Decided not to use normalized advantages here,
        #   is this correct? (it's how it's done in Kostrikov's)
        action_loss = -(
            cast(torch.FloatTensor, batch["adv_targ"]).detach() * action_log_probs
        )

        if self.acktr:
            # TODO: Currently acktr doesn't really work because of this natural gradient stuff
            #   that we should figure out how to integrate properly.
            get_logger().warning("acktr is only partially supported.")

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
        **kwargs,
    ):
        losses_per_step = self.loss_per_step(
            step_count=step_count, batch=batch, actor_critic_output=actor_critic_output,
        )
        losses = {
            key: (loss.mean(), weight)
            for (key, (loss, weight)) in losses_per_step.items()
        }

        total_loss = cast(
            torch.Tensor,
            sum(
                loss * weight if weight is not None else loss
                for loss, weight in losses.values()
            ),
        )

        return (
            total_loss,
            {
                self.loss_key: total_loss.item(),
                **{key: loss.item() for key, (loss, _) in losses.items()},
            },
        )


class A2C(A2CACKTR):
    """A2C Loss."""

    def __init__(
        self,
        value_loss_coef,
        entropy_coef,
        entropy_method_name: str = "entropy",
        *args,
        **kwargs,
    ):
        super().__init__(
            value_loss_coef=value_loss_coef,
            entropy_coef=entropy_coef,
            acktr=False,
            entropy_method_name=entropy_method_name,
            *args,
            **kwargs,
        )


class ACKTR(A2CACKTR):
    """ACKTR Loss.

    This code is not supported as it currently lacks an implementation
    for recurrent models.
    """

    def __init__(
        self,
        value_loss_coef,
        entropy_coef,
        entropy_method_name: str = "entropy",
        *args,
        **kwargs,
    ):
        super().__init__(
            value_loss_coef=value_loss_coef,
            entropy_coef=entropy_coef,
            acktr=True,
            entropy_method_name=entropy_method_name,
            *args,
            **kwargs,
        )


A2CConfig = dict(value_loss_coef=0.5, entropy_coef=0.01,)
