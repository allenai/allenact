"""Implementation of A2C and ACKTR losses."""
import typing
import warnings
from typing import Tuple, Dict, Union

import torch

from onpolicy_sync.losses.abstract_loss import AbstractActorCriticLoss
from rl_base.common import ActorCriticOutput
from rl_base.distributions import CategoricalDistr


class A2CACKTR(AbstractActorCriticLoss):
    """Class implementing A2C and ACKTR losses.

    # Attributes

    acktr : `True` if should use ACKTR loss (currently not supported), otherwise uses A2C loss.
    value_loss_coef : Weight of value loss.
    entropy_coef : Weight of entropy (encouraging) loss.
    """

    def __init__(self, value_loss_coef, entropy_coef, acktr=False, *args, **kwargs):
        """Initializer.

        See class documentation for parameter definitions.
        """
        super().__init__(*args, **kwargs)
        self.acktr = acktr

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

    def loss(  # type: ignore
        self,
        step_count: int,
        batch: Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]],
        actor_critic_output: ActorCriticOutput[CategoricalDistr],
        *args,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Dict[str, float]]:
        actions = typing.cast(torch.LongTensor, batch["actions"])
        values = actor_critic_output.values
        action_log_probs = actor_critic_output.distributions.log_probs(actions)

        dist_entropy: torch.FloatTensor = actor_critic_output.distributions.entropy().mean()

        value_loss = (
            0.5
            * (typing.cast(torch.FloatTensor, batch["returns"]) - values).pow(2).mean()
        )

        # TODO: Decided not to use normalized advantages here, is this correct? (it's how it's done in Kostrikov's)
        action_loss = -(
            typing.cast(torch.FloatTensor, batch["adv_targ"]).detach()
            * action_log_probs
        ).mean()

        if self.acktr:
            warnings.warn("acktr is only partially supported.")
        # TODO: Currently acktr doesn't really work because of this natural gradient stuff
        #   that we should figure out how to integrate properly.
        # if self.acktr and self.optimizer.steps % self.optimizer.Ts == 0:
        #     # Sampled fisher, see Martens 2014
        #     self.actor_critic.zero_grad()
        #     pg_fisher_loss = -action_log_probs.mean()
        #
        #     value_noise = torch.randn(values.size())
        #     if values.is_cuda:
        #         value_noise = value_noise.cuda()
        #
        #     sample_values = values + value_noise
        #     vf_fisher_loss = -(values - sample_values.detach()).pow(2).mean()
        #
        #     fisher_loss = pg_fisher_loss + vf_fisher_loss
        #     self.optimizer.acc_stats = True
        #     fisher_loss.backward(retain_graph=True)
        #     self.optimizer.acc_stats = False
        #
        # if self.acktr == False:
        #     nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)

        total_loss = (
            value_loss * self.value_loss_coef
            + action_loss
            - dist_entropy * self.entropy_coef
        )

        return (
            total_loss,
            {
                "total": total_loss.item(),
                "value": value_loss.item(),
                "action": action_loss.item(),
                "entropy": -dist_entropy.item(),
            },
        )


class A2C(A2CACKTR):
    """A2C Loss."""

    def __init__(self, value_loss_coef, entropy_coef, *args, **kwargs):
        super().__init__(
            value_loss_coef=value_loss_coef, entropy_coef=entropy_coef, acktr=False,
        )


class ACKTR(A2CACKTR):
    """ACKTR Loss. This code is not supported as it currently lacks an implementation for recurrent models."""

    def __init__(self, value_loss_coef, entropy_coef, *args, **kwargs):
        super().__init__(
            value_loss_coef=value_loss_coef, entropy_coef=entropy_coef, acktr=True,
        )


A2CConfig = dict(value_loss_coef=0.5, entropy_coef=0.01,)
