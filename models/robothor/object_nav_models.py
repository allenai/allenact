import logging

import torch.nn as nn
import torch
import gym
from gym.spaces.dict import Dict as SpaceDict

from rl_base.common import ActorCriticOutput
from models.basic_models import RNNStateEncoder
from onpolicy_sync.policy import LinearCriticHead, LinearActorHead
from models.object_nav_models import ObjectNavBaselineActorCritic
from .basic_models import RobothorTargetImTensorProcessor


class RobothorObjectNavActorCritic(ObjectNavBaselineActorCritic):
    def __init__(
        self,
        action_space: gym.spaces.Discrete,
        observation_space: SpaceDict,
        goal_sensor_uuid: str,
        hidden_size=512,
        object_type_embedding_dim=32,
        encoder_output_dims=1568,
    ):
        super().__init__(
            action_space=action_space,
            goal_sensor_uuid=goal_sensor_uuid,
            observation_space=observation_space,
        )

        self.goal_sensor_uuid = goal_sensor_uuid
        self._n_object_types = self.observation_space.spaces[self.goal_sensor_uuid].n
        self._hidden_size = hidden_size
        self.object_type_embedding_size = object_type_embedding_dim

        self.visual_encoder = RobothorTargetImTensorProcessor(
            self.observation_space,
            self.goal_sensor_uuid,
            encoder_output_dims,
            self.object_type_embedding_size,
        )

        self.state_encoder = RNNStateEncoder(
            encoder_output_dims, self.recurrent_hidden_state_size,
        )

        self.actor = LinearActorHead(self.recurrent_hidden_state_size, action_space.n)
        self.critic = LinearCriticHead(self.recurrent_hidden_state_size)

        self.object_type_embedding = nn.Embedding(
            num_embeddings=self._n_object_types,
            embedding_dim=object_type_embedding_dim,
        )

        self.train()

    def forward(self, observations, rnn_hidden_states, prev_actions, masks):
        if not self.is_blind:
            perception_embed = self.visual_encoder(observations)
            x = perception_embed

        x, rnn_hidden_states = self.state_encoder(x, rnn_hidden_states, masks)

        return (
            ActorCriticOutput(
                distributions=self.actor(x), values=self.critic(x), extras={}
            ),
            rnn_hidden_states,
        )
