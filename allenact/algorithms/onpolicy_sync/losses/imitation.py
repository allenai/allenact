"""Defining imitation losses for actor critic type models."""

from typing import Dict, cast, Optional
from collections import OrderedDict

import torch

from allenact.algorithms.onpolicy_sync.losses.abstract_loss import (
    AbstractActorCriticLoss,
    ObservationType,
)
from allenact.base_abstractions.distributions import (
    Distr,
    CategoricalDistr,
    SequentialDistr,
    ConditionalDistr,
)
from allenact.base_abstractions.misc import ActorCriticOutput
from allenact.base_abstractions.sensor import AbstractExpertSensor
import allenact.utils.spaces_utils as su


class Imitation(AbstractActorCriticLoss):
    """Expert imitation loss."""

    def __init__(
        self, expert_sensor: Optional[AbstractExpertSensor] = None, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.expert_sensor = expert_sensor

    def group_loss(
        self,
        distribution: CategoricalDistr,
        expert_actions: torch.Tensor,
        expert_actions_masks: torch.Tensor,
    ):
        assert isinstance(distribution, CategoricalDistr) or (
            isinstance(distribution, ConditionalDistr)
            and isinstance(distribution.distr, CategoricalDistr)
        ), "This implementation only supports (groups of) `CategoricalDistr`"

        expert_successes = expert_actions_masks.sum()

        log_probs = distribution.log_prob(cast(torch.LongTensor, expert_actions))
        assert (
            log_probs.shape[: len(expert_actions_masks.shape)]
            == expert_actions_masks.shape
        )

        # Add dimensions to `expert_actions_masks` on the right to allow for masking
        # if necessary.
        len_diff = len(log_probs.shape) - len(expert_actions_masks.shape)
        assert len_diff >= 0
        expert_actions_masks = expert_actions_masks.view(
            *expert_actions_masks.shape, *((1,) * len_diff)
        )

        group_loss = -(expert_actions_masks * log_probs).sum() / torch.clamp(
            expert_successes, min=1
        )

        return group_loss, expert_successes

    def loss(  # type: ignore
        self,
        step_count: int,
        batch: ObservationType,
        actor_critic_output: ActorCriticOutput[Distr],
        *args,
        **kwargs,
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
        observations = cast(Dict[str, torch.Tensor], batch["observations"])

        losses = OrderedDict()

        should_report_loss = False

        if "expert_action" in observations:
            if self.expert_sensor is None or not self.expert_sensor.use_groups:
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

                total_loss, expert_successes = self.group_loss(
                    cast(CategoricalDistr, actor_critic_output.distributions),
                    expert_actions,
                    expert_actions_masks,
                )

                should_report_loss = expert_successes.item() != 0
            else:
                expert_actions = su.unflatten(
                    self.expert_sensor.observation_space, observations["expert_action"]
                )

                total_loss = 0

                ready_actions = OrderedDict()

                for group_name, cd in zip(
                    self.expert_sensor.group_spaces,
                    cast(
                        SequentialDistr, actor_critic_output.distributions
                    ).conditional_distrs,
                ):
                    assert group_name == cd.action_group_name

                    cd.reset()
                    cd.condition_on_input(**ready_actions)

                    expert_action = expert_actions[group_name][
                        AbstractExpertSensor.ACTION_POLICY_LABEL
                    ]
                    expert_action_masks = expert_actions[group_name][
                        AbstractExpertSensor.EXPERT_SUCCESS_LABEL
                    ]

                    ready_actions[group_name] = expert_action

                    current_loss, expert_successes = self.group_loss(
                        cd, expert_action, expert_action_masks,
                    )

                    should_report_loss = (
                        expert_successes.item() != 0 or should_report_loss
                    )

                    cd.reset()

                    if expert_successes.item() != 0:
                        losses[group_name + "_cross_entropy"] = current_loss.item()
                        total_loss = total_loss + current_loss
        elif "expert_policy" in observations:
            if self.expert_sensor is None or not self.expert_sensor.use_groups:
                assert isinstance(
                    actor_critic_output.distributions, CategoricalDistr
                ), "This implementation currently only supports `CategoricalDistr`"

                expert_policies = cast(Dict[str, torch.Tensor], batch["observations"])[
                    "expert_policy"
                ][..., :-1]
                expert_actions_masks = cast(
                    Dict[str, torch.Tensor], batch["observations"]
                )["expert_policy"][..., -1:]

                expert_successes = expert_actions_masks.sum()
                if expert_successes.item() > 0:
                    should_report_loss = True

                log_probs = cast(
                    CategoricalDistr, actor_critic_output.distributions
                ).log_probs_tensor

                # Add dimensions to `expert_actions_masks` on the right to allow for masking
                # if necessary.
                len_diff = len(log_probs.shape) - len(expert_actions_masks.shape)
                assert len_diff >= 0
                expert_actions_masks = expert_actions_masks.view(
                    *expert_actions_masks.shape, *((1,) * len_diff)
                )

                total_loss = (
                    -(log_probs * expert_policies) * expert_actions_masks
                ).sum() / torch.clamp(expert_successes, min=1)
            else:
                raise NotImplementedError(
                    "This implementation currently only supports `CategoricalDistr`"
                )
        else:
            raise NotImplementedError(
                "Imitation loss requires either `expert_action` or `expert_policy`"
                " sensor to be active."
            )
        return (
            total_loss,
            {"expert_cross_entropy": total_loss.item(), **losses}
            if should_report_loss
            else {},
        )
