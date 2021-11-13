"""Baseline models for use in the object navigation task.

Object navigation is currently available as a Task in AI2-THOR and
Facebook's Habitat.
"""
from typing import Tuple, Dict, Optional, cast

import gym
import torch
import torch.nn as nn
from gym.spaces.dict import Dict as SpaceDict

from allenact.algorithms.onpolicy_sync.policy import (
    ActorCriticModel,
    LinearCriticHead,
    LinearActorHead,
    DistributionType,
    Memory,
    ObservationType,
)
from allenact.base_abstractions.distributions import CategoricalDistr
from allenact.base_abstractions.misc import ActorCriticOutput
from allenact.embodiedai.models.basic_models import RNNStateEncoder


class CLIPObjectNavActorCritic(ActorCriticModel[CategoricalDistr]):
    def __init__(
        self,
        action_space: gym.spaces.Discrete,
        observation_space: SpaceDict,
        goal_sensor_uuid: str,
        rgb_resnet_preprocessor_uuid: str,
        hidden_size: int = 512,
        goal_dims: int = 32,
        resnet_compressor_hidden_out_dims: Tuple[int, int] = (128, 32),
        combiner_hidden_out_dims: Tuple[int, int] = (128, 32),
        include_auxiliary_head: bool = False,
    ):

        super().__init__(
            action_space=action_space, observation_space=observation_space,
        )

        self._hidden_size = hidden_size
        self.include_auxiliary_head = include_auxiliary_head

        resnet_preprocessor_uuid = rgb_resnet_preprocessor_uuid

        self.encoder = CLIPActorCriticEncoder(
            self.observation_space,
            goal_sensor_uuid,
            resnet_preprocessor_uuid,
            goal_dims,
            resnet_compressor_hidden_out_dims,
            combiner_hidden_out_dims,
            self._hidden_size
        )

        self.actor = LinearActorHead(self._hidden_size, action_space.n)
        self.critic = LinearCriticHead(self._hidden_size)
        if self.include_auxiliary_head:
            self.auxiliary_actor = LinearActorHead(self._hidden_size, action_space.n)

        self.train()

    @property
    def recurrent_hidden_state_size(self) -> int:
        """The recurrent hidden state size of the model."""
        return self._hidden_size

    @property
    def is_blind(self) -> bool:
        """True if the model is blind (e.g. neither 'depth' or 'rgb' is an
        input observation type)."""
        return False

    @property
    def num_recurrent_layers(self) -> int:
        """Number of recurrent hidden layers."""
        return self.encoder.state_encoder.num_recurrent_layers

    def _recurrent_memory_specification(self):
        return dict(
            rnn=(
                (
                    ("layer", self.num_recurrent_layers),
                    ("sampler", None),
                    ("hidden", self.recurrent_hidden_state_size),
                ),
                torch.float32,
            )
        )

    def get_object_type_encoding(
        self, observations: Dict[str, torch.FloatTensor]
    ) -> torch.FloatTensor:
        """Get the object type encoding from input batched observations."""
        return self.encoder.get_object_type_encoding(observations)

    def forward(  # type:ignore
        self,
        observations: ObservationType,
        memory: Memory,
        prev_actions: torch.Tensor,
        masks: torch.FloatTensor,
    ) -> Tuple[ActorCriticOutput[DistributionType], Optional[Memory]]:
        x, rnn_hidden_states = self.encoder(observations, memory, masks)
        return (
            ActorCriticOutput(
                distributions=self.actor(x),
                values=self.critic(x),
                extras={"auxiliary_distributions": self.auxiliary_actor(x)}
                if self.include_auxiliary_head
                else {},
            ),
            memory.set_tensor("rnn", rnn_hidden_states),
        )


class CLIPActorCriticEncoder(nn.Module):
    def __init__(
        self,
        observation_spaces: SpaceDict,
        goal_sensor_uuid: str,
        resnet_preprocessor_uuid: str,
        class_dims: int = 32,
        resnet_compressor_hidden_out_dims: Tuple[int, int] = (128, 32),
        combiner_hidden_out_dims: Tuple[int, int] = (128, 32),
        rnn_hidden_size: int = 512
    ) -> None:
        super().__init__()
        self.goal_uuid = goal_sensor_uuid
        self.resnet_uuid = resnet_preprocessor_uuid
        self.class_dims = class_dims
        self.resnet_hid_out_dims = resnet_compressor_hidden_out_dims
        self.combine_hid_out_dims = combiner_hidden_out_dims

        self.embed_class = nn.Linear(1024, self.class_dims)

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

        self.state_encoder = RNNStateEncoder(self.output_dims, rnn_hidden_size,)

    @property
    def is_blind(self):
        return False

    @property
    def output_dims(self):
        return (
            self.combine_hid_out_dims[-1]
            * self.resnet_tensor_shape[1]
            * self.resnet_tensor_shape[2]
        )

    def get_object_type_encoding(
        self, observations: Dict[str, torch.FloatTensor]
    ) -> torch.FloatTensor:
        """Get the object type encoding from input batched observations."""
        return cast(
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

    def adapt_input(self, observations):
        resnet = observations[self.resnet_uuid]
        goal = observations[self.goal_uuid]

        use_agent = False
        nagent = 1

        if len(resnet.shape) == 6:
            use_agent = True
            nstep, nsampler, nagent = resnet.shape[:3]
        else:
            nstep, nsampler = resnet.shape[:2]

        observations[self.resnet_uuid] = resnet.view(-1, *resnet.shape[-3:])
        observations[self.goal_uuid] = goal.view(-1, goal.shape[-1])

        return observations, use_agent, nstep, nsampler, nagent

    @staticmethod
    def adapt_output(x, use_agent, nstep, nsampler, nagent):
        if use_agent:
            return x.view(nstep, nsampler, nagent, -1)
        return x.view(nstep, nsampler * nagent, -1)

    def forward(self, observations, memory: Memory, masks: torch.FloatTensor):
        observations, use_agent, nstep, nsampler, nagent = self.adapt_input(
            observations
        )

        embs = [
            self.compress_resnet(observations),
            self.distribute_target(observations),
        ]
        x = self.target_obs_combiner(torch.cat(embs, dim=1,))
        x = x.reshape(x.size(0), -1)  # flatten

        x = self.adapt_output(x, use_agent, nstep, nsampler, nagent)

        x, rnn_hidden_states = self.state_encoder(x, memory.tensor("rnn"), masks)

        return x, rnn_hidden_states
