"""Baseline models for use in the point navigation task.

Object navigation is currently available as a Task in AI2-THOR and
Facebook's Habitat.
"""
from typing import Optional, List, Union

import gym
import torch
from gym.spaces import Dict as SpaceDict
from torch import nn as nn

from allenact.algorithms.onpolicy_sync.policy import ObservationType
from allenact.embodiedai.models import resnet as resnet
from allenact.embodiedai.models.basic_models import SimpleCNN
from allenact.embodiedai.models.visual_nav_models import (
    VisualNavActorCritic,
    FusionType,
)


class PointNavActorCritic(VisualNavActorCritic):
    """Use raw image as observation to the agent."""

    def __init__(
        # base params
        self,
        action_space: gym.spaces.Discrete,
        observation_space: SpaceDict,
        goal_sensor_uuid: str,
        hidden_size=512,
        num_rnn_layers=1,
        rnn_type="GRU",
        add_prev_actions=False,
        action_embed_size=4,
        multiple_beliefs=False,
        beliefs_fusion: Optional[FusionType] = None,
        auxiliary_uuids: Optional[List[str]] = None,
        # custom params
        rgb_uuid: Optional[str] = None,
        depth_uuid: Optional[str] = None,
        embed_coordinates=False,
        coordinate_embedding_dim=8,
        coordinate_dims=2,
        # perception backbone params,
        backbone="gnresnet18",
        resnet_baseplanes=32,
    ):
        super().__init__(
            action_space=action_space,
            observation_space=observation_space,
            hidden_size=hidden_size,
            multiple_beliefs=multiple_beliefs,
            beliefs_fusion=beliefs_fusion,
            auxiliary_uuids=auxiliary_uuids,
        )

        self.goal_sensor_uuid = goal_sensor_uuid
        self.embed_coordinates = embed_coordinates
        if self.embed_coordinates:
            self.coordinate_embedding_size = coordinate_embedding_dim
        else:
            self.coordinate_embedding_size = coordinate_dims

        self.sensor_fusion = False
        if rgb_uuid is not None and depth_uuid is not None:
            self.sensor_fuser = nn.Linear(hidden_size * 2, hidden_size)
            self.sensor_fusion = True

        self.backbone = backbone
        if backbone == "simple_cnn":
            self.visual_encoder = SimpleCNN(
                observation_space=observation_space,
                output_size=hidden_size,
                rgb_uuid=rgb_uuid,
                depth_uuid=depth_uuid,
            )
        else:  # resnet family
            self.visual_encoder = resnet.GroupNormResNetEncoder(
                observation_space=observation_space,
                output_size=hidden_size,
                rgb_uuid=rgb_uuid,
                depth_uuid=depth_uuid,
                baseplanes=resnet_baseplanes,
                ngroups=resnet_baseplanes // 2,
                make_backbone=getattr(resnet, backbone),
            )

        if self.embed_coordinates:
            self.coordinate_embedding = nn.Linear(
                coordinate_dims, coordinate_embedding_dim
            )

        self.create_state_encoders(
            obs_embed_size=self.goal_visual_encoder_output_dims,
            num_rnn_layers=num_rnn_layers,
            rnn_type=rnn_type,
            add_prev_actions=add_prev_actions,
            prev_action_embed_size=action_embed_size,
        )

        self.create_actorcritic_head()

        self.create_aux_models(
            obs_embed_size=self.goal_visual_encoder_output_dims,
            action_embed_size=action_embed_size,
        )

        self.train()

    @property
    def is_blind(self):
        return self.visual_encoder.is_blind

    @property
    def goal_visual_encoder_output_dims(self):
        dims = self.coordinate_embedding_size
        if self.is_blind:
            return dims
        return dims + self.recurrent_hidden_state_size

    def get_target_coordinates_encoding(self, observations):
        if self.embed_coordinates:
            return self.coordinate_embedding(
                observations[self.goal_sensor_uuid].to(torch.float32)
            )
        else:
            return observations[self.goal_sensor_uuid].to(torch.float32)

    def forward_encoder(self, observations: ObservationType) -> torch.FloatTensor:
        target_encoding = self.get_target_coordinates_encoding(observations)
        obs_embeds: Union[torch.Tensor, List[torch.Tensor]]
        obs_embeds = [target_encoding]

        if not self.is_blind:
            perception_embed = self.visual_encoder(observations)
            if self.sensor_fusion:
                perception_embed = self.sensor_fuser(perception_embed)
            obs_embeds = [perception_embed] + obs_embeds

        obs_embeds = torch.cat(obs_embeds, dim=-1)
        return obs_embeds
