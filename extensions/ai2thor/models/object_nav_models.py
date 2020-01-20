import gym

from extensions.ai2thor.models.basic_models import SimpleCNN, RNNStateEncoder
from onpolicy_sync.policy import ActorCriticModel, LinearCriticHead, LinearActorHead
import torch.nn as nn
import torch

from rl_base.common import ActorCriticOutput
from rl_base.distributions import CategoricalDistr
from gym.spaces.dict import Dict as SpaceDict


class ObjectNavBaselineActorCritic(ActorCriticModel[CategoricalDistr]):
    def __init__(
        self,
        action_space: gym.spaces.Discrete,
        observation_space: SpaceDict,
        goal_sensor_uuid: str,
        hidden_size=512,
        object_type_embedding_dim=8,
    ):
        super().__init__(action_space=action_space, observation_space=observation_space)

        self.goal_sensor_uuid = goal_sensor_uuid
        self._n_object_types = self.observation_space.spaces[self.goal_sensor_uuid].n
        self.recurrent_hidden_state_size = hidden_size
        self.object_type_embedding_size = object_type_embedding_dim

        self.visual_encoder = SimpleCNN(self.observation_space, hidden_size)

        self.state_encoder = RNNStateEncoder(
            (0 if self.is_blind else self.recurrent_hidden_state_size)
            + object_type_embedding_dim,
            self.recurrent_hidden_state_size,
        )

        self.actor = LinearActorHead(
            self.recurrent_hidden_state_size, action_space.n
        )
        self.critic = LinearCriticHead(
            self.recurrent_hidden_state_size
        )

        self.object_type_embedding = nn.Embedding(
            num_embeddings=self._n_object_types,
            embedding_dim=object_type_embedding_dim,
        )

        self.train()

    @property
    def output_size(self):
        return self.recurrent_hidden_state_size

    @property
    def is_blind(self):
        return self.visual_encoder.is_blind

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    def get_object_type_encoding(self, observations):
        return self.object_type_embedding(observations[self.goal_sensor_uuid])

    def forward(self, observations, rnn_hidden_states, prev_actions, masks):
        target_encoding = self.get_object_type_encoding(observations)
        x = [target_encoding]

        if not self.is_blind:
            perception_embed = self.visual_encoder(observations)
            x = [perception_embed] + x

        x = torch.cat(x, dim=1)
        x, rnn_hidden_states = self.state_encoder(x, rnn_hidden_states, masks)

        return (
            ActorCriticOutput(
                distributions=self.actor(x), values=self.critic(x), extras={}
            ),
            rnn_hidden_states,
        )
