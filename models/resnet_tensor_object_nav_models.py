from typing import Tuple, Dict

import gym
from gym.spaces import Dict as SpaceDict
import torch
import torch.nn as nn

from models.basic_models import RNNStateEncoder
from onpolicy_sync.policy import ActorCriticModel, LinearActorHead, LinearCriticHead
from rl_base.common import ActorCriticOutput
from rl_base.distributions import CategoricalDistr


class ResnetTensorObjectNavActorCritic(ActorCriticModel[CategoricalDistr]):
    def __init__(
        self,
        action_space: gym.spaces.Discrete,
        observation_space: SpaceDict,
        goal_sensor_uuid: str,
        rnn_hidden_size=512,
        goal_dims: int = 32,
        compressor_hidden_out_dims: Tuple[int, int] = (128, 32),
        combiner_hidden_out_dims: Tuple[int, int] = (128, 32),
    ):
        super().__init__(
            action_space=action_space, observation_space=observation_space,
        )

        self.goal_sensor_uuid = goal_sensor_uuid
        self.hidden_size = rnn_hidden_size

        self.goal_imtensor_encoder = ResnetTensorGoalEncoder(
            self.observation_space,
            self.goal_sensor_uuid,
            goal_dims,
            compressor_hidden_out_dims,
            combiner_hidden_out_dims,
        )

        self.state_encoder = RNNStateEncoder(
            self.goal_imtensor_encoder.output_dims, rnn_hidden_size,
        )

        self.actor = LinearActorHead(self.hidden_size, action_space.n)
        self.critic = LinearCriticHead(self.hidden_size)

        self.train()

    @property
    def recurrent_hidden_state_size(self) -> int:
        """The recurrent hidden state size of the model."""
        return self.hidden_size

    @property
    def is_blind(self) -> bool:
        """True if the model is blind (e.g. neither 'depth' or 'rgb' is an
        input observation type)."""
        return self.goal_imtensor_encoder.is_blind

    @property
    def num_recurrent_layers(self) -> int:
        """Number of recurrent hidden layers."""
        return self.state_encoder.num_recurrent_layers

    def get_object_type_encoding(
        self, observations: Dict[str, torch.FloatTensor]
    ) -> torch.FloatTensor:
        """Get the object type encoding from input batched observations."""
        return self.goal_imtensor_encoder.get_object_type_encoding(observations)

    def forward(self, observations, rnn_hidden_states, prev_actions, masks):
        x = self.goal_imtensor_encoder(observations)
        x, rnn_hidden_states = self.state_encoder(x, rnn_hidden_states, masks)

        return (
            ActorCriticOutput(
                distributions=self.actor(x), values=self.critic(x), extras={}
            ),
            rnn_hidden_states,
        )


class ResnetTensorGoalEncoder(nn.Module):
    def __init__(
        self,
        observation_spaces: SpaceDict,
        goal_sensor_uuid: str,
        goal_dims: int = 32,
        compressor_hidden_out_dims: Tuple[int, int] = (128, 32),
        combiner_hidden_out_dims: Tuple[int, int] = (128, 32),
    ) -> None:
        super().__init__()

        self.goal_sensor_uuid = goal_sensor_uuid
        self.goal_dims = goal_dims
        self.compress_hid_out_dims = compressor_hidden_out_dims
        self.combine_hid_out_dims = combiner_hidden_out_dims

        self.embed_goal = nn.Embedding(
            num_embeddings=observation_spaces.spaces[self.goal_sensor_uuid].n,
            embedding_dim=self.goal_dims,
        )

        self.blind = "rgb_resnet" not in observation_spaces.spaces

        if not self.blind:
            self.in_tensor_shape = observation_spaces.spaces["rgb_resnet"].shape
            print(self.in_tensor_shape)

            self.imtensor_compressor = nn.Sequential(
                nn.Conv2d(self.in_tensor_shape[0], self.compress_hid_out_dims[0], 1),
                nn.ReLU(),
                nn.Conv2d(*self.compress_hid_out_dims[0:2], 1),
                nn.ReLU(),
            )

            self.target_imtensor_combiner = nn.Sequential(
                nn.Conv2d(
                    self.compress_hid_out_dims[1] + self.goal_dims,
                    self.combine_hid_out_dims[0],
                    1,
                ),
                nn.ReLU(),
                nn.Conv2d(*self.combine_hid_out_dims[0:2], 1),
            )

    @property
    def is_blind(self):
        return self.blind

    @property
    def output_dims(self):
        if self.blind:
            return self.goal_dims
        else:
            return (
                self.combine_hid_out_dims[-1]
                * self.in_tensor_shape[1]
                * self.in_tensor_shape[2]
            )

    def get_object_type_encoding(
        self, observations: Dict[str, torch.FloatTensor]
    ) -> torch.FloatTensor:
        """Get the object type encoding from input batched observations."""
        return self.embed_goal(observations[self.goal_sensor_uuid].to(torch.int64))

    def forward(self, observations):
        target_emb = self.embed_goal(observations[self.goal_sensor_uuid])

        if self.blind:
            return target_emb

        imtensor = self.imtensor_compressor(observations["rgb_resnet"])

        x = self.target_imtensor_combiner(
            torch.cat(
                (
                    imtensor,
                    target_emb.view(-1, self.goal_dims, 1, 1).expand(
                        -1, -1, imtensor.shape[-2], imtensor.shape[-1]
                    ),
                ),
                dim=-3,
            )
        )
        return x.view(x.size(0), -1)  # flatten
