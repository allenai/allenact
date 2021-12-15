# Original work Copyright (c) Facebook, Inc. and its affiliates.
# Modified work Copyright (c) Allen Institute for AI
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Several of the models defined in this file are modified versions of those found in 
    https://github.com/joel99/habitat-pointnav-aux/blob/master/habitat_baselines/
"""


import torch
import torch.nn as nn

from allenact.utils.model_utils import FeatureEmbedding
from allenact.embodiedai.aux_losses.losses import (
    InverseDynamicsLoss,
    TemporalDistanceLoss,
    CPCALoss,
)


class AuxiliaryModel(nn.Module):
    """
    The class of defining the models for all kinds of self-supervised auxiliary tasks
    """

    def __init__(
        self,
        aux_uuid: str,
        action_dim: int,
        obs_embed_dim: int,
        belief_dim: int,
        action_embed_size: int = 4,
        cpca_classifier_hidden_dim: int = 32,
    ):
        super().__init__()
        self.aux_uuid = aux_uuid
        self.action_dim = action_dim
        self.obs_embed_dim = obs_embed_dim
        self.belief_dim = belief_dim

        if self.aux_uuid == InverseDynamicsLoss.UUID:
            self.decoder = nn.Linear(
                2 * self.obs_embed_dim + self.belief_dim, self.action_dim
            )
        elif self.aux_uuid == TemporalDistanceLoss.UUID:
            self.decoder = nn.Linear(2 * self.obs_embed_dim + self.belief_dim, 1)
        elif CPCALoss.UUID in self.aux_uuid:  # the CPCA family with various k
            ## Auto-regressive model to predict future context
            self.action_embedder = FeatureEmbedding(
                self.action_dim + 1, action_embed_size
            )
            # NOTE: add extra 1 in embedding dict cuz we will pad zero actions?
            self.context_model = nn.GRU(action_embed_size, self.belief_dim)

            ## Classifier to estimate mutual information
            self.classifier = nn.Sequential(
                nn.Linear(
                    self.belief_dim + self.obs_embed_dim, cpca_classifier_hidden_dim
                ),
                nn.ReLU(),
                nn.Linear(cpca_classifier_hidden_dim, 1),
            )

        else:
            raise ValueError("Unknown Auxiliary Loss UUID")

    def forward(self, features: torch.FloatTensor):
        if self.aux_uuid in [InverseDynamicsLoss.UUID, TemporalDistanceLoss.UUID]:
            return self.decoder(features)
        else:
            raise NotImplementedError(
                f"Auxiliary model with UUID {self.aux_uuid} does not support `forward` call."
            )
