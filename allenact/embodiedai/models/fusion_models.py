# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# Adapted from https://github.com/joel99/habitat-pointnav-aux/blob/master/habitat_baselines/

from typing import Callable, Tuple, Dict, Optional, Union, List, cast, Any
from allenact.utils.system import get_logger

import math
import torch
import torch.nn as nn


class Fusion(nn.Module):
    def __init__(self, hidden_size, obs_embed_size, num_tasks):
        super().__init__()
        self.hidden_size = hidden_size  # H
        self.obs_embed_size = obs_embed_size  # Z
        self.num_tasks = num_tasks  # k

    def forward(
        self,
        total_beliefs: torch.FloatTensor,  # (T, N, H, K)
        obs_embeds: torch.FloatTensor,  # (T, N, Z)
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:  # (T, N, H), (T, N, K)

        num_steps, num_samplers, _, _ = total_beliefs.shape
        total_beliefs = total_beliefs.view(
            num_steps * num_samplers, self.hidden_size, self.num_tasks
        )
        obs_embeds = obs_embeds.view(num_steps * num_samplers, -1)

        weights = self.get_belief_weights(
            total_beliefs=total_beliefs,  # (T*N, H, K)
            obs_embeds=obs_embeds,  # (T*N, Z)
        ).unsqueeze(
            -1
        )  # (T*N, K, 1)

        beliefs = torch.bmm(total_beliefs, weights)  # (T*N, H, 1)

        beliefs = beliefs.squeeze(-1).view(num_steps, num_samplers, self.hidden_size)
        weights = weights.squeeze(-1).view(num_steps, num_samplers, self.num_tasks)

        return beliefs, weights

    def get_belief_weights(
        self,
        total_beliefs: torch.FloatTensor,  # (T*N, H, K)
        obs_embeds: torch.FloatTensor,  # (T*N, Z)
    ) -> torch.FloatTensor:  # (T*N, K)
        raise NotImplementedError()


class AverageFusion(Fusion):
    UUID = "avg"

    def get_belief_weights(
        self,
        total_beliefs: torch.FloatTensor,  # (T*N, H, K)
        obs_embeds: torch.FloatTensor,  # (T*N, Z)
    ) -> torch.FloatTensor:  # (T*N, K)

        batch_size = total_beliefs.shape[0]
        weights = torch.ones(batch_size, self.num_tasks).to(total_beliefs)
        weights /= self.num_tasks
        return weights


class SoftmaxFusion(Fusion):
    UUID = "smax"
    """
    Situational Fusion of Visual Representation for Visual Navigation
    https://arxiv.org/abs/1908.09073
    """

    def __init__(self, hidden_size, obs_embed_size, num_tasks):
        super().__init__(hidden_size, obs_embed_size, num_tasks)
        # mapping from rnn input to task
        # ignore beliefs
        self.linear = nn.Linear(obs_embed_size, num_tasks)

    def get_belief_weights(
        self,
        total_beliefs: torch.FloatTensor,  # (T*N, H, K)
        obs_embeds: torch.FloatTensor,  # (T*N, Z)
    ) -> torch.FloatTensor:  # (T*N, K)

        scores = self.linear(obs_embeds)  # (T*N, K)
        weights = torch.softmax(scores, dim=-1)
        return weights


class AttentiveFusion(Fusion):
    UUID = "attn"
    """
    Attention is All You Need
    https://arxiv.org/abs/1706.03762
    i.e. scaled dot-product attention
    """

    def __init__(self, hidden_size, obs_embed_size, num_tasks):
        super().__init__(hidden_size, obs_embed_size, num_tasks)
        self.linear = nn.Linear(obs_embed_size, hidden_size)

    def get_belief_weights(
        self,
        total_beliefs: torch.FloatTensor,  # (T*N, H, K)
        obs_embeds: torch.FloatTensor,  # (T*N, Z)
    ) -> torch.FloatTensor:  # (T*N, K)

        queries = self.linear(obs_embeds).unsqueeze(1)  # (T*N, 1, H)
        scores = torch.bmm(queries, total_beliefs).squeeze(1)  # (T*N, K)
        weights = torch.softmax(
            scores / math.sqrt(self.hidden_size), dim=-1
        )  # (T*N, K)
        return weights


FusionModels = {
    AverageFusion.UUID: AverageFusion,
    SoftmaxFusion.UUID: SoftmaxFusion,
    AttentiveFusion.UUID: AttentiveFusion,
}
