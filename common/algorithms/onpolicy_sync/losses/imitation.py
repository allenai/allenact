"""Defining imitation losses for actor critic type models."""

import typing
from typing import Dict, Union

import torch

from common.algorithms.onpolicy_sync.losses.abstract_loss import AbstractActorCriticLoss
from common.rl_base.common import ActorCriticOutput
from common.rl_base.distributions import CategoricalDistr


class Imitation(AbstractActorCriticLoss):
    """Expert imitation loss."""

    def loss(  # type: ignore
        self,
        step_count: int,
        batch: Dict[str, Union[Dict[str, torch.Tensor], torch.Tensor]],
        actor_critic_output: ActorCriticOutput[CategoricalDistr],
        *args,
        **kwargs
    ):
        """Computes the imitation loss.

        # Parameters

        batch : A batch of data corresponding to the information collected when rolling out (possibly many) agents
            over a fixed number of steps. In particular this batch should have the same format as that returned by
            `RolloutStorage.recurrent_generator`.
            Here `batch["observations"]` must contain `"expert_action"` observations
            or `"expert_policy"` observations. See `ExpertActionSensor` (or `ExpertPolicySensor`) for an example of
            a sensor producing such observations.
        actor_critic_output : The output of calling an ActorCriticModel on the observations in `batch`.
        args : Extra args. Ignored.
        kwargs : Extra kwargs. Ignored.

        # Returns

        A (0-dimensional) torch.FloatTensor corresponding to the computed loss. `.backward()` will be called on this
        tensor in order to compute a gradient update to the ActorCriticModel's parameters.
        """
        observations = typing.cast(Dict[str, torch.Tensor], batch["observations"])
        if "expert_action" in observations:
            expert_actions_and_mask = observations["expert_action"]
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
                * actor_critic_output.distributions.log_probs(
                    typing.cast(torch.LongTensor, expert_actions)
                )
            ).sum() / torch.clamp(expert_successes, min=1)
        elif "expert_policy" in observations:
            expert_policies = typing.cast(
                Dict[str, torch.Tensor], batch["observations"]
            )["expert_policy"][:, :-1]
            expert_actions_masks = typing.cast(
                Dict[str, torch.Tensor], batch["observations"]
            )["expert_policy"][:, -1:]

            expert_successes = expert_actions_masks.sum()
            if expert_successes.item() == 0:
                return 0, {}

            total_loss = (
                -(actor_critic_output.distributions.log_probs_tensor * expert_policies)
                * expert_actions_masks
            ).sum() / expert_successes
        else:
            raise NotImplementedError(
                "Imitation loss requires either `expert_action` or `expert_policy`"
                " sensor to be active."
            )

        return (
            total_loss,
            {"expert_cross_entropy": total_loss.item(),},
        )
