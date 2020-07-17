import typing
from typing import Tuple, Dict

import gym
from gym.spaces import Dict as SpaceDict
import torch
import torch.nn as nn

from models.basic_models import RNNStateEncoder
from onpolicy_sync.policy import ActorCriticModel, LinearActorHead, LinearCriticHead
from rl_base.common import ActorCriticOutput, Memory
from rl_base.distributions import CategoricalDistr
from utils.system import LOGGER


class ResnetTensorObjectNavActorCritic(ActorCriticModel[CategoricalDistr]):
    def __init__(
        self,
        action_space: gym.spaces.Discrete,
        observation_space: SpaceDict,
        goal_sensor_uuid: str,
        resnet_preprocessor_uuid: str,
        rnn_hidden_size: int = 512,
        goal_dims: int = 32,
        resnet_compressor_hidden_out_dims: Tuple[int, int] = (128, 32),
        combiner_hidden_out_dims: Tuple[int, int] = (128, 32),
    ):
        super().__init__(
            action_space=action_space, observation_space=observation_space,
        )

        self.hidden_size = rnn_hidden_size

        self.goal_visual_encoder = ResnetTensorGoalEncoder(
            self.observation_space,
            goal_sensor_uuid,
            resnet_preprocessor_uuid,
            goal_dims,
            resnet_compressor_hidden_out_dims,
            combiner_hidden_out_dims,
        )

        self.state_encoder = RNNStateEncoder(
            self.goal_visual_encoder.output_dims, rnn_hidden_size,
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
        return self.goal_visual_encoder.is_blind

    @property
    def num_recurrent_layers(self) -> int:
        """Number of recurrent hidden layers."""
        return self.state_encoder.num_recurrent_layers

    def get_object_type_encoding(
        self, observations: Dict[str, torch.FloatTensor]
    ) -> torch.FloatTensor:
        """Get the object type encoding from input batched observations."""
        return self.goal_visual_encoder.get_object_type_encoding(observations)

    def forward(self, observations, rnn_hidden_states, prev_actions, masks):
        x = self.goal_visual_encoder(observations)
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
        resnet_preprocessor_uuid: str,
        class_dims: int = 32,
        resnet_compressor_hidden_out_dims: Tuple[int, int] = (128, 32),
        combiner_hidden_out_dims: Tuple[int, int] = (128, 32),
    ) -> None:
        super().__init__()

        self.goal_uuid = goal_sensor_uuid
        self.resnet_uuid = resnet_preprocessor_uuid

        self.class_dims = class_dims

        self.resnet_hid_out_dims = resnet_compressor_hidden_out_dims
        self.combine_hid_out_dims = combiner_hidden_out_dims

        self.embed_class = nn.Embedding(
            num_embeddings=observation_spaces.spaces[self.goal_uuid].n,
            embedding_dim=self.class_dims,
        )

        self.blind = self.resnet_uuid not in observation_spaces.spaces

        if not self.blind:
            self.resnet_tensor_shape = observation_spaces.spaces[self.resnet_uuid].shape

            self.resnet_compressor = nn.Sequential(
                nn.Conv2d(self.resnet_tensor_shape[0], self.resnet_hid_out_dims[0], 1),
                nn.ReLU(),
                nn.Conv2d(*self.resnet_hid_out_dims[0:2], 1),
                nn.ReLU(),
            )

            self.target_obs_combiner = nn.Sequential(
                nn.Conv2d(
                    self.resnet_hid_out_dims[1] + self.class_dims,
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
            return self.class_dims
        else:
            return (
                self.combine_hid_out_dims[-1]
                * self.resnet_tensor_shape[1]
                * self.resnet_tensor_shape[2]
            )

    def get_object_type_encoding(
        self, observations: Dict[str, torch.FloatTensor]
    ) -> torch.FloatTensor:
        """Get the object type encoding from input batched observations."""
        return typing.cast(
            torch.FloatTensor,
            self.embed_class(observations[self.goal_uuid].to(torch.int64)),
        )

    def compress_resnet(self, observations):
        return self.resnet_compressor(observations[self.resnet_uuid])

    def distribute_target(self, observations):
        target_emb = self.embed_class(observations[self.goal_uuid])
        return target_emb.view(-1, self.class_dims, 1, 1).expand(
            -1, -1, self.resnet_tensor_shape[-2], self.resnet_tensor_shape[-1]
        )

    def forward(self, observations):
        if self.blind:
            return self.embed_class(observations[self.goal_uuid])

        embs = [
            self.compress_resnet(observations),
            self.distribute_target(observations),
        ]

        x = self.target_obs_combiner(torch.cat(embs, dim=-3,))

        return x.view(x.size(0), -1)  # flatten


class ResnetTensorObjectNavActorCriticMemory(ResnetTensorObjectNavActorCritic):
    @property
    def recurrent_hidden_state_size(
        self,
    ) -> Dict[str, Tuple[Tuple[int, ...], int, torch.dtype]]:
        """The memory spec of the model: A dictionary with string keys and tuple values, each with the dimensions of the
        memory, e.g. (2, 32) for two layers of 32-dimensional recurrent hidden states; an integer indicating the index
        of the sampler in a batch, e.g. 1 for RNNs; the data type, e.g. torch.float32.
        """
        return {
            "rnn_hidden": (
                (self.state_encoder.num_recurrent_layers, self.hidden_size),
                1,
                torch.float32,
            )
        }

    @property
    def num_recurrent_layers(self) -> int:
        """Returns -1, indicating we are using a memory specification in recurrent_hidden_state_size."""
        return -1

    def get_object_type_encoding(
        self, observations: Dict[str, torch.FloatTensor]
    ) -> torch.FloatTensor:
        """Get the object type encoding from input batched observations."""
        return self.goal_visual_encoder.get_object_type_encoding(observations)

    def forward(self, observations, rnn_hidden_states, prev_actions, masks):
        x = self.goal_visual_encoder(observations)

        x, mem_return = self.state_encoder(
            x, rnn_hidden_states.tensor("rnn_hidden"), masks
        )
        rnn_hidden_states["rnn_hidden"] = (
            mem_return,
            rnn_hidden_states.sampler_dim("rnn_hidden"),
        )

        return (
            ActorCriticOutput(
                distributions=self.actor(x), values=self.critic(x), extras={}
            ),
            rnn_hidden_states,
        )
