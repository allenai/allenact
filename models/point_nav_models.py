import gym

from models.basic_models import SimpleCNN, RNNStateEncoder
from onpolicy_sync.policy import ActorCriticModel, LinearCriticHead, LinearActorHead
import torch.nn as nn
import torch

from rl_base.common import ActorCriticOutput
from rl_base.distributions import CategoricalDistr
from gym.spaces.dict import Dict as SpaceDict


class PointNavActorCriticSimpleConv(ActorCriticModel[CategoricalDistr]):
    def __init__(
        self,
        action_space: gym.spaces.Discrete,
        observation_space: SpaceDict,
        goal_sensor_uuid: str,
        hidden_size=512,
        embed_coordinates=False,
        coordinate_embedding_dim=8,
        coordinate_dims=2
    ):
        super().__init__(action_space=action_space, observation_space=observation_space)

        self.goal_sensor_uuid = goal_sensor_uuid
        self.recurrent_hidden_state_size = hidden_size
        self.embed_coordinates = embed_coordinates
        if self.embed_coordinates:
            self.coorinate_embedding_size = coordinate_embedding_dim
        else:
            self.coorinate_embedding_size = coordinate_dims

        self.visual_encoder = SimpleCNN(observation_space, hidden_size)

        self.state_encoder = RNNStateEncoder(
            (0 if self.is_blind else self.recurrent_hidden_state_size)
            + self.coorinate_embedding_size,
            self.recurrent_hidden_state_size,
            num_layers=2,
            rnn_type="LSTM"
        )

        self.actor = LinearActorHead(
            self.recurrent_hidden_state_size, action_space.n
        )
        self.critic = LinearCriticHead(
            self.recurrent_hidden_state_size
        )

        if self.embed_coordinates:
            self.coordinate_embedding = nn.Linear(coordinate_dims, coordinate_embedding_dim)

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

    def get_target_coordinates_encoding(self, observations):
        if self.embed_coordinates:
            return self.coordinate_embedding(
                observations[self.goal_sensor_uuid].to(torch.float32)
            )
        else:
            return observations[self.goal_sensor_uuid].to(torch.float32)

    def recurrent_hidden_state_size(self):
        return self._hidden_size

    def forward(self, observations, rnn_hidden_states, prev_actions, masks):
        target_encoding = self.get_target_coordinates_encoding(observations)
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


class PointNavActorCriticResNet50(ActorCriticModel[CategoricalDistr]):
    def __init__(
        self,
        action_space: gym.spaces.Discrete,
        observation_space: SpaceDict,
        goal_sensor_uuid: str,
        hidden_size=512,
        embed_coordinates=False,
        coordinate_embedding_dim=8,
        coordinate_dims=2
    ):
        super().__init__(action_space=action_space, observation_space=observation_space)

        self.goal_sensor_uuid = goal_sensor_uuid
        self.embed_coordinates = embed_coordinates
        self.recurrent_hidden_state_size = hidden_size + coordinate_dims
        if self.embed_coordinates:
            self.coorinate_embedding_size = coordinate_embedding_dim
        else:
            self.coorinate_embedding_size = coordinate_dims

        if 'rgb' in observation_space.spaces and 'depth' in observation_space.spaces:
            self.visual_encoder = nn.Linear(4096, hidden_size)
        else:
            self.visual_encoder = nn.Linear(2048, hidden_size)

        self.actor = LinearActorHead(
            self.recurrent_hidden_state_size, action_space.n
        )
        self.critic = LinearCriticHead(
            self.recurrent_hidden_state_size
        )

        if self.embed_coordinates:
            self.coordinate_embedding = nn.Linear(coordinate_dims, coordinate_embedding_dim)

        self.train()

    @property
    def output_size(self):
        return self.recurrent_hidden_state_size

    @property
    def is_blind(self):
        return False

    @property
    def num_recurrent_layers(self):
        return 0

    def get_target_coordinates_encoding(self, observations):
        if self.embed_coordinates:
            return self.coordinate_embedding(
                observations[self.goal_sensor_uuid].to(torch.float32)
            )
        else:
            return observations[self.goal_sensor_uuid].to(torch.float32)

    def recurrent_hidden_state_size(self):
        return self._hidden_size

    def forward(self, observations, rnn_hidden_states, prev_actions, masks):
        target_encoding = self.get_target_coordinates_encoding(observations)

        x = [target_encoding]

        embs = []
        if "rgb" in observations:
            embs.append(observations["rgb"].view(-1, observations["rgb"].shape[-1]))
        if "depth" in observations:
            embs.append(observations["depth"].view(-1, observations["depth"].shape[-1]))
        emb = torch.cat(embs, dim=1)

        x = [self.visual_encoder(emb)] + x

        x = torch.cat(x, dim=1)

        return (
            ActorCriticOutput(
                distributions=self.actor(x), values=self.critic(x), extras={}
            ),
            torch.zeros((1, x.shape[0], self.recurrent_hidden_state_size))
        )


class PointNavActorCriticResNet50GRU(ActorCriticModel[CategoricalDistr]):
    def __init__(
        self,
        action_space: gym.spaces.Discrete,
        observation_space: SpaceDict,
        goal_sensor_uuid: str,
        hidden_size=512,
        embed_coordinates=False,
        coordinate_embedding_dim=8,
        coordinate_dims=2,
    ):
        super().__init__(action_space=action_space, observation_space=observation_space)

        self.goal_sensor_uuid = goal_sensor_uuid
        self.recurrent_hidden_state_size = hidden_size
        self.embed_coordinates = embed_coordinates
        if self.embed_coordinates:
            self.coorinate_embedding_size = coordinate_embedding_dim
        else:
            self.coorinate_embedding_size = coordinate_dims

        if 'rgb' in observation_space.spaces and 'depth' in observation_space.spaces:
            self.visual_encoder = nn.Linear(4096, hidden_size)
        else:
            self.visual_encoder = nn.Linear(2048, hidden_size)

        self.state_encoder = RNNStateEncoder(
            (0 if self.is_blind else self.recurrent_hidden_state_size)
            + self.coorinate_embedding_size,
            self.recurrent_hidden_state_size,
        )

        self.actor = LinearActorHead(
            self.recurrent_hidden_state_size, action_space.n
        )
        self.critic = LinearCriticHead(
            self.recurrent_hidden_state_size
        )

        if self.embed_coordinates:
            self.coordinate_embedding = nn.Linear(coordinate_dims, coordinate_embedding_dim)

        self.train()

    @property
    def output_size(self):
        return self.recurrent_hidden_state_size

    @property
    def is_blind(self):
        return False

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    def get_target_coordinates_encoding(self, observations):
        if self.embed_coordinates:
            return self.coordinate_embedding(
                observations[self.goal_sensor_uuid].to(torch.float32)
            )
        else:
            return observations[self.goal_sensor_uuid].to(torch.float32)

    def recurrent_hidden_state_size(self):
        return self._hidden_size

    def forward(self, observations, rnn_hidden_states, prev_actions, masks):
        target_encoding = self.get_target_coordinates_encoding(observations)
        x = [target_encoding]

        embs = []
        if "rgb" in observations:
            embs.append(observations["rgb"].view(-1, observations["rgb"].shape[-1]))
        if "rgb_resnet" in observations:
            embs.append(observations["rgb_resnet"].view(-1, observations["rgb_resnet"].shape[-1]))
        if "depth" in observations:
            embs.append(observations["depth"].view(-1, observations["depth"].shape[-1]))
        emb = torch.cat(embs, dim=1)

        x = [self.visual_encoder(emb)] + x
        x = torch.cat(x, dim=1)

        x, rnn_hidden_states = self.state_encoder(x, rnn_hidden_states, masks)

        return (
            ActorCriticOutput(
                distributions=self.actor(x), values=self.critic(x), extras={}
            ),
            rnn_hidden_states,
        )