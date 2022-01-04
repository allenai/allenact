# Original work Copyright (c) Facebook, Inc. and its affiliates.
# Modified work Copyright (c) Allen Institute for AI
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Defining the auxiliary loss for actor critic type models.

Several of the losses defined in this file are modified versions of those found in
    https://github.com/joel99/habitat-pointnav-aux/blob/master/habitat_baselines/
"""


from typing import Dict, cast, Tuple, List
import abc

import numpy as np
import torch
import torch.nn as nn

from allenact.algorithms.onpolicy_sync.losses.abstract_loss import (
    AbstractActorCriticLoss,
    ObservationType,
)
from allenact.base_abstractions.distributions import CategoricalDistr
from allenact.base_abstractions.misc import ActorCriticOutput


def _bernoulli_subsample_mask_like(masks, p=0.1):
    return (torch.rand_like(masks) <= p).float()


class MultiAuxTaskNegEntropyLoss(AbstractActorCriticLoss):
    """Used in multiple auxiliary tasks setting.

    Add a negative entropy loss over all the task weights.
    """

    UUID = "multitask_entropy"  # make sure this is unique

    def __init__(self, task_names: List[str], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_tasks = len(task_names)
        self.task_names = task_names

    def loss(  # type: ignore
        self,
        step_count: int,
        batch: ObservationType,
        actor_critic_output: ActorCriticOutput[CategoricalDistr],
        *args,
        **kwargs
    ) -> Tuple[torch.FloatTensor, Dict[str, float]]:
        task_weights = actor_critic_output.extras[self.UUID]
        task_weights = task_weights.view(-1, self.num_tasks)
        entropy = CategoricalDistr(task_weights).entropy()

        avg_loss = (-entropy).mean()
        avg_task_weights = task_weights.mean(dim=0)  # (K)

        outputs = {"entropy_loss": cast(torch.Tensor, avg_loss).item()}
        for i in range(self.num_tasks):
            outputs["weight_" + self.task_names[i]] = cast(
                torch.Tensor, avg_task_weights[i]
            ).item()

        return (
            avg_loss,
            outputs,
        )


class AuxiliaryLoss(AbstractActorCriticLoss):
    """Base class of auxiliary loss.

    Any auxiliary task loss should inherit from it, and implement
    the `get_aux_loss` function.
    """

    def __init__(self, auxiliary_uuid: str, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.auxiliary_uuid = auxiliary_uuid

    def loss(  # type: ignore
        self,
        step_count: int,
        batch: ObservationType,
        actor_critic_output: ActorCriticOutput[CategoricalDistr],
        *args,
        **kwargs
    ) -> Tuple[torch.FloatTensor, Dict[str, float]]:

        # auxiliary loss
        return self.get_aux_loss(
            **actor_critic_output.extras[self.auxiliary_uuid],
            observations=batch["observations"],
            actions=batch["actions"],
            masks=batch["masks"],
        )

    @abc.abstractmethod
    def get_aux_loss(
        self,
        aux_model: nn.Module,
        observations: ObservationType,
        obs_embeds: torch.FloatTensor,
        actions: torch.FloatTensor,
        beliefs: torch.FloatTensor,
        masks: torch.FloatTensor,
        *args,
        **kwargs
    ):
        raise NotImplementedError()


def _propagate_final_beliefs_to_all_steps(
    beliefs: torch.Tensor, masks: torch.Tensor, num_sampler: int, num_steps: int,
):
    final_beliefs = torch.zeros_like(beliefs)  # (T, B, *)
    start_locs_list = []
    end_locs_list = []

    for i in range(num_sampler):
        # right shift: to locate the 1 before 0 and ignore the 1st element
        end_locs = torch.where(masks[1:, i] == 0)[0]  # maybe [], dtype=torch.Long

        start_locs = torch.cat(
            [torch.tensor([0]).to(end_locs), end_locs + 1]
        )  # add the first element
        start_locs_list.append(start_locs)

        end_locs = torch.cat(
            [end_locs, torch.tensor([num_steps - 1]).to(end_locs)]
        )  # add the last element
        end_locs_list.append(end_locs)

        for st, ed in zip(start_locs, end_locs):
            final_beliefs[st : ed + 1, i] = beliefs[ed, i]

    return final_beliefs, start_locs_list, end_locs_list


class InverseDynamicsLoss(AuxiliaryLoss):
    """Auxiliary task of Inverse Dynamics from Auxiliary Tasks Speed Up
    Learning PointGoal Navigation (Ye, 2020) https://arxiv.org/abs/2007.04561
    originally from Curiosity-driven Exploration by Self-supervised Prediction
    (Pathak, 2017) https://arxiv.org/abs/1705.05363."""

    UUID = "InvDyn"

    def __init__(
        self, subsample_rate: float = 0.2, subsample_min_num: int = 10, *args, **kwargs
    ):
        """Subsample the valid samples by the rate of `subsample_rate`, if the
        total num of the valid samples is larger than `subsample_min_num`."""
        super().__init__(auxiliary_uuid=self.UUID, *args, **kwargs)

        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction="none")
        self.subsample_rate = subsample_rate
        self.subsample_min_num = subsample_min_num

    def get_aux_loss(
        self,
        aux_model: nn.Module,
        observations: ObservationType,
        obs_embeds: torch.FloatTensor,
        actions: torch.FloatTensor,
        beliefs: torch.FloatTensor,
        masks: torch.FloatTensor,
        *args,
        **kwargs
    ):
        ## we discard the last action in the batch
        num_steps, num_sampler = actions.shape  # T, B
        actions = cast(torch.LongTensor, actions)
        actions = actions[:-1]  # (T-1, B)

        ## find the final belief state based on masks
        # we did not compute loss here as model.forward is compute-heavy
        masks = masks.squeeze(-1)  # (T, B)

        final_beliefs, _, _ = _propagate_final_beliefs_to_all_steps(
            beliefs, masks, num_sampler, num_steps,
        )

        ## compute CE loss
        decoder_in = torch.cat(
            [obs_embeds[:-1], obs_embeds[1:], final_beliefs[:-1]], dim=2
        )  # (T-1, B, *)

        preds = aux_model(decoder_in)  # (T-1, B, A)
        # cross entropy loss require class dim at 1
        loss = self.cross_entropy_loss(
            preds.view((num_steps - 1) * num_sampler, -1),  # ((T-1)*B, A)
            actions.flatten(),  #  ((T-1)*B,)
        )
        loss = loss.view(num_steps - 1, num_sampler)  # (T-1, B)

        # def vanilla_valid_losses(loss, num_sampler, end_locs_batch):
        #     ##  this is just used to verify the vectorized version works correctly.
        #     ##  not used for experimentation
        #     valid_losses = []
        #     for i in range(num_sampler):
        #         end_locs = end_locs_batch[i]
        #         for j in range(len(end_locs)):
        #             if j == 0:
        #                 start_loc = 0
        #             else:
        #                 start_loc = end_locs[j - 1] + 1
        #             end_loc = end_locs[j]
        #             if end_loc - start_loc <= 0:  # the episode only 1-step
        #                 continue
        #             valid_losses.append(loss[start_loc:end_loc, i])

        #     if len(valid_losses) == 0:
        #         valid_losses = torch.zeros(1, dtype=torch.float).to(loss)
        #     else:
        #         valid_losses = torch.cat(valid_losses)  # (sum m, )
        #     return valid_losses

        # valid_losses = masks[1:] * loss # (T-1, B)
        # valid_losses0 = vanilla_valid_losses(loss, num_sampler, end_locs_batch)
        # assert valid_losses0.sum() == valid_losses.sum()

        num_valid_losses = torch.count_nonzero(masks[1:])
        if num_valid_losses < self.subsample_min_num:  # don't subsample
            subsample_rate = 1.0
        else:
            subsample_rate = self.subsample_rate

        loss_masks = masks[1:] * _bernoulli_subsample_mask_like(
            masks[1:], subsample_rate
        )
        num_valid_losses = torch.count_nonzero(loss_masks)
        avg_loss = (loss * loss_masks).sum() / torch.clamp(num_valid_losses, min=1.0)

        return (
            avg_loss,
            {"total": cast(torch.Tensor, avg_loss).item(),},
        )


class TemporalDistanceLoss(AuxiliaryLoss):
    """Auxiliary task of Temporal Distance from Auxiliary Tasks Speed Up
    Learning PointGoal Navigation (Ye, 2020)
    https://arxiv.org/abs/2007.04561."""

    UUID = "TempDist"

    def __init__(self, num_pairs: int = 8, epsiode_len_min: int = 5, *args, **kwargs):
        super().__init__(auxiliary_uuid=self.UUID, *args, **kwargs)
        self.num_pairs = num_pairs
        self.epsiode_len_min = float(epsiode_len_min)

    def get_aux_loss(
        self,
        aux_model: nn.Module,
        observations: ObservationType,
        obs_embeds: torch.FloatTensor,
        actions: torch.FloatTensor,
        beliefs: torch.FloatTensor,
        masks: torch.FloatTensor,
        *args,
        **kwargs
    ):
        ## we discard the last action in the batch
        num_steps, num_sampler = actions.shape  # T, B

        ## find the final belief state based on masks
        # we did not compute loss here as model.forward is compute-heavy
        masks = masks.squeeze(-1)  # (T, B)

        (
            final_beliefs,
            start_locs_list,
            end_locs_list,
        ) = _propagate_final_beliefs_to_all_steps(
            beliefs, masks, num_sampler, num_steps,
        )

        ## also find the locs_batch of shape (M, 3)
        # the last dim: [0] is on num_sampler loc, [1] and [2] is start and end locs
        # of one episode
        # in other words: at locs_batch[m, 0] in num_sampler dim, there exists one episode
        # starting from locs_batch[m, 1], ends at locs_batch[m, 2] (included)
        locs_batch = []
        for i in range(num_sampler):
            locs_batch.append(
                torch.stack(
                    [
                        i * torch.ones_like(start_locs_list[i]),
                        start_locs_list[i],
                        end_locs_list[i],
                    ],
                    dim=-1,
                )
            )  # shape (M[i], 3)
        locs_batch = torch.cat(locs_batch)  # shape (M, 3)

        temporal_dist_max = (
            locs_batch[:, 2] - locs_batch[:, 1]
        ).float()  # end - start, (M)
        # create normalizer that ignores too short episode, otherwise 1/T
        normalizer = torch.where(
            temporal_dist_max > self.epsiode_len_min,
            1.0 / temporal_dist_max,
            torch.tensor([0]).to(temporal_dist_max),
        )  # (M)

        # sample valid pairs: sampled_pairs shape (M, num_pairs, 3)
        # where M is the num of total episodes in the batch
        locs = locs_batch.cpu().numpy()  # as torch.randint only support int, not tensor
        sampled_pairs = np.random.randint(
            np.repeat(locs[:, [1]], 2 * self.num_pairs, axis=-1),  # (M, 2*k)
            np.repeat(locs[:, [2]] + 1, 2 * self.num_pairs, axis=-1),  # (M, 2*k)
        ).reshape(
            -1, self.num_pairs, 2
        )  # (M, k, 2)
        sampled_pairs_batch = torch.from_numpy(sampled_pairs).to(
            locs_batch
        )  # (M, k, 2)

        num_sampler_batch = locs_batch[:, [0]].expand(
            -1, 2 * self.num_pairs
        )  # (M, 1) -> (M, 2*k)
        num_sampler_batch = num_sampler_batch.reshape(
            -1, self.num_pairs, 2
        )  # (M, k, 2)

        sampled_obs_embeds = obs_embeds[
            sampled_pairs_batch, num_sampler_batch
        ]  # (M, k, 2, H1)
        sampled_final_beliefs = final_beliefs[
            sampled_pairs_batch, num_sampler_batch
        ]  # (M, k, 2, H2)
        features = torch.cat(
            [
                sampled_obs_embeds[:, :, 0],
                sampled_obs_embeds[:, :, 1],
                sampled_final_beliefs[:, :, 0],
            ],
            dim=-1,
        )  # (M, k, 2*H1 + H2)

        pred_temp_dist = aux_model(features).squeeze(-1)  # (M, k)
        true_temp_dist = (
            sampled_pairs_batch[:, :, 1] - sampled_pairs_batch[:, :, 0]
        ).float()  # (M, k)

        pred_error = (pred_temp_dist - true_temp_dist) * normalizer.unsqueeze(1)
        loss = 0.5 * (pred_error).pow(2)
        avg_loss = loss.mean()

        return (
            avg_loss,
            {"total": cast(torch.Tensor, avg_loss).item(),},
        )


class CPCALoss(AuxiliaryLoss):
    """Auxiliary task of CPC|A from Auxiliary Tasks Speed Up Learning PointGoal
    Navigation (Ye, 2020) https://arxiv.org/abs/2007.04561 originally from
    Neural Predictive Belief Representations (Guo, 2018)
    https://arxiv.org/abs/1811.06407."""

    UUID = "CPCA"

    def __init__(
        self, planning_steps: int = 8, subsample_rate: float = 0.2, *args, **kwargs
    ):
        super().__init__(auxiliary_uuid=self.UUID, *args, **kwargs)
        self.planning_steps = planning_steps
        self.subsample_rate = subsample_rate
        self.cross_entropy_loss = nn.BCEWithLogitsLoss(reduction="none")

    def get_aux_loss(
        self,
        aux_model: nn.Module,
        observations: ObservationType,
        obs_embeds: torch.FloatTensor,
        actions: torch.FloatTensor,
        beliefs: torch.FloatTensor,
        masks: torch.FloatTensor,
        *args,
        **kwargs
    ):
        # prepare for autoregressive inputs: c_{t+1:t+k} = GRU(b_t, a_{t:t+k-1}) <-> z_{t+k}
        ## where b_t = RNN(b_{t-1}, z_t, a_{t-1}), prev action is optional
        num_steps, num_sampler, obs_embed_size = obs_embeds.shape  # T, N, H_O
        assert 0 < self.planning_steps <= num_steps

        ## prepare positive and negatives that sample from all the batch
        positives = obs_embeds  # (T, N, -1)
        negative_inds = torch.randperm(num_steps * num_sampler).to(positives.device)
        negatives = torch.gather(  # input[index[i,j]][j]
            positives.view(num_steps * num_sampler, -1),
            dim=0,
            index=negative_inds.view(num_steps * num_sampler, 1).expand(
                num_steps * num_sampler, positives.size(-1)
            ),
        ).view(
            num_steps, num_sampler, -1
        )  # (T, N, -1)

        ## prepare action sequences and initial beliefs
        action_embedding = aux_model.action_embedder(actions)  # (T, N, -1)
        action_embed_size = action_embedding.size(-1)
        action_padding = torch.zeros(
            self.planning_steps - 1, num_sampler, action_embed_size
        ).to(
            action_embedding
        )  # (k-1, N, -1)
        action_padded = torch.cat(
            (action_embedding, action_padding), dim=0
        )  # (T+k-1, N, -1)
        ## unfold function will create consecutive action sequences
        action_seq = (
            action_padded.unfold(dimension=0, size=self.planning_steps, step=1)
            .permute(3, 0, 1, 2)
            .view(self.planning_steps, num_steps * num_sampler, action_embed_size)
        )  # (k, T*N, -1)
        beliefs = beliefs.view(num_steps * num_sampler, -1).unsqueeze(0)  # (1, T*N, -1)

        # get future contexts c_{t+1:t+k} = GRU(b_t, a_{t:t+k-1})
        future_contexts_all, _ = aux_model.context_model(
            action_seq, beliefs
        )  # (k, T*N, -1)
        ## NOTE: future_contexts_all starting from next step t+1 to t+k, not t to t+k-1
        future_contexts_all = future_contexts_all.view(
            self.planning_steps, num_steps, num_sampler, -1
        ).permute(
            1, 0, 2, 3
        )  # (k, T, N, -1)

        # get all the classifier scores I(c_{t+1:t+k}; z_{t+1:t+k})
        positives_padding = torch.zeros(
            self.planning_steps, num_sampler, obs_embed_size
        ).to(
            positives
        )  # (k, N, -1)
        positives_padded = torch.cat(
            (positives[1:], positives_padding), dim=0
        )  # (T+k-1, N, -1)
        positives_expanded = positives_padded.unfold(
            dimension=0, size=self.planning_steps, step=1
        ).permute(
            0, 3, 1, 2
        )  # (T, k, N, -1)
        positives_logits = aux_model.classifier(
            torch.cat([positives_expanded, future_contexts_all], -1)
        )  # (T, k, N, 1)
        positive_loss = self.cross_entropy_loss(
            positives_logits, torch.ones_like(positives_logits)
        )  # (T, k, N, 1)

        negatives_padding = torch.zeros(
            self.planning_steps, num_sampler, obs_embed_size
        ).to(
            negatives
        )  # (k, N, -1)
        negatives_padded = torch.cat(
            (negatives[1:], negatives_padding), dim=0
        )  # (T+k-1, N, -1)
        negatives_expanded = negatives_padded.unfold(
            dimension=0, size=self.planning_steps, step=1
        ).permute(
            0, 3, 1, 2
        )  # (T, k, N, -1)
        negatives_logits = aux_model.classifier(
            torch.cat([negatives_expanded, future_contexts_all], -1)
        )  # (T, k, N, 1)
        negative_loss = self.cross_entropy_loss(
            negatives_logits, torch.zeros_like(negatives_logits)
        )  # (T, k, N, 1)

        # Masking to get valid scores
        ## masks: Note which timesteps [1, T+k+1] could have valid queries, at distance (k) (note offset by 1)
        ## we will extract the **diagonals** as valid_masks from masks later as below
        ## the vertical axis is (absolute) real timesteps, the horizontal axis is (relative) planning timesteps
        ## | - - - - - |
        ## | .         |
        ## | , .       |
        ## | . , .     |
        ## | , . , .   |
        ## |   , . , . |
        ## |     , . , |
        ## |       , . |
        ## |         , |
        ## | - - - - - |
        masks = masks.squeeze(-1)  # (T, N)
        pred_masks = torch.ones(
            num_steps + self.planning_steps,
            self.planning_steps,
            num_sampler,
            1,
            dtype=torch.bool,
        ).to(
            beliefs.device
        )  # (T+k, k, N, 1)

        pred_masks[
            num_steps - 1 :
        ] = False  # GRU(b_t, a_{t:t+k-1}) is invalid when t >= T, as we don't have real z_{t+1}
        for j in range(1, self.planning_steps + 1):  # for j-step predictions
            pred_masks[
                : j - 1, j - 1
            ] = False  # Remove the upper triangle above the diagnonal (but I think this is unnecessary for valid_masks)
            for n in range(num_sampler):
                has_zeros_batch = torch.where(masks[:, n] == 0)[0]
                # in j-step prediction, timesteps z -> z + j are disallowed as those are the first j timesteps of a new episode
                # z-> z-1 because of pred_masks being offset by 1
                for z in has_zeros_batch:
                    pred_masks[
                        z - 1 : z - 1 + j, j - 1, n
                    ] = False  # can affect j timesteps

        # instead of the whole range, we actually are only comparing a window i:i+k for each query/target i - for each, select the appropriate k
        # we essentially gather diagonals from this full mask, t of them, k long
        valid_diagonals = [
            torch.diagonal(pred_masks, offset=-i) for i in range(num_steps)
        ]  # pull the appropriate k per timestep
        valid_masks = (
            torch.stack(valid_diagonals, dim=0).permute(0, 3, 1, 2).float()
        )  # (T, N, 1, k) -> (T, k, N, 1)
        # print(valid_masks.int().squeeze(-1)); print(masks) # verify its correctness

        loss_masks = valid_masks * _bernoulli_subsample_mask_like(
            valid_masks, self.subsample_rate
        )  # (T, k, N, 1)
        num_valid_losses = torch.count_nonzero(loss_masks)
        avg_positive_loss = (positive_loss * loss_masks).sum() / torch.clamp(
            num_valid_losses, min=1.0
        )
        avg_negative_loss = (negative_loss * loss_masks).sum() / torch.clamp(
            num_valid_losses, min=1.0
        )

        avg_loss = avg_positive_loss + avg_negative_loss

        return (
            avg_loss,
            {
                "total": cast(torch.Tensor, avg_loss).item(),
                "positive_loss": cast(torch.Tensor, avg_positive_loss).item(),
                "negative_loss": cast(torch.Tensor, avg_negative_loss).item(),
            },
        )


class CPCA1Loss(CPCALoss):
    UUID = "CPCA_1"

    def __init__(self, subsample_rate: float = 0.2, *args, **kwargs):
        super().__init__(
            planning_steps=1, subsample_rate=subsample_rate, *args, **kwargs
        )


class CPCA2Loss(CPCALoss):
    UUID = "CPCA_2"

    def __init__(self, subsample_rate: float = 0.2, *args, **kwargs):
        super().__init__(
            planning_steps=2, subsample_rate=subsample_rate, *args, **kwargs
        )


class CPCA4Loss(CPCALoss):
    UUID = "CPCA_4"

    def __init__(self, subsample_rate: float = 0.2, *args, **kwargs):
        super().__init__(
            planning_steps=4, subsample_rate=subsample_rate, *args, **kwargs
        )


class CPCA8Loss(CPCALoss):
    UUID = "CPCA_8"

    def __init__(self, subsample_rate: float = 0.2, *args, **kwargs):
        super().__init__(
            planning_steps=8, subsample_rate=subsample_rate, *args, **kwargs
        )


class CPCA16Loss(CPCALoss):
    UUID = "CPCA_16"

    def __init__(self, subsample_rate: float = 0.2, *args, **kwargs):
        super().__init__(
            planning_steps=16, subsample_rate=subsample_rate, *args, **kwargs
        )
