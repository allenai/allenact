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
        include_auxiliary_head: bool = False,
    ):

        super().__init__(action_space=action_space, observation_space=observation_space,)

        self._hidden_size = hidden_size
        self.include_auxiliary_head = include_auxiliary_head

        self.encoder = CLIPActorCriticEncoder(
            self.observation_space,
            goal_sensor_uuid,
            rgb_resnet_preprocessor_uuid,
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
        rnn_hidden_size: int = 512
    ) -> None:
        super().__init__()
        self.goal_uuid = goal_sensor_uuid
        self.resnet_uuid = resnet_preprocessor_uuid

        self.state_encoder = RNNStateEncoder(self.output_dims, rnn_hidden_size,)

    @property
    def is_blind(self):
        return False

    @property
    def output_dims(self):
        return (1024)

    def get_object_type_encoding(
        self, observations: Dict[str, torch.FloatTensor]
    ) -> torch.FloatTensor:
        """Get the object type encoding from input batched observations."""
        return observations[self.goal_uuid]

    def forward(self, observations, memory: Memory, masks: torch.FloatTensor):
        # observations are (nstep, nsampler, nagent, *features) or (nstep, nsampler, *features)
        # rnn encoder input should be (*n..., vector)
        # function output should be (*n..., vector)

        nstep, nsampler = observations[self.resnet_uuid].shape[:2]

        x, rnn_hidden_states = self.state_encoder(
            observations[self.resnet_uuid],
            memory.tensor("rnn"),
            masks
        )

        x = x.view(nstep * nsampler, -1)

        # adapt input
        resnet_obs = observations[self.resnet_uuid].view(nstep * nsampler, -1)
        goal_obs = observations[self.goal_uuid].view(nstep * nsampler, -1)

        # nn layers
        output = (resnet_obs + x) * goal_obs

        # adapt output
        output = output.view(nstep, nsampler, -1)

        return output, rnn_hidden_states
