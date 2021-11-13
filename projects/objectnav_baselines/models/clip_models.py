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

        self.v_proj = nn.Sequential(
            nn.Conv2d(2048, 1024, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True)
        )

        self.l_proj = nn.Linear(1024, 1024)

        self.fusion = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=0, bias=False),
            nn.GroupNorm(32, 512),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            #
            nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0, bias=False),
            nn.GroupNorm(32, 512),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            #
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )

        self.state_encoder = RNNStateEncoder(self.output_dims, rnn_hidden_size,)

    @property
    def is_blind(self):
        return False

    @property
    def output_dims(self):
        return (512)

    def get_object_type_encoding(
        self, observations: Dict[str, torch.FloatTensor]
    ) -> torch.FloatTensor:
        """Get the object type encoding from input batched observations."""
        return observations[self.goal_uuid]

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

        x = self.v_proj(observations[self.resnet_uuid])

        l = self.l_proj(observations[self.goal_uuid])
        l = l.unsqueeze(-1).unsqueeze(-1)
        l = l.repeat(1, 1, x.shape[-2], x.shape[-1])

        fused = self.fusion(x * l)
        output = self.adapt_output(fused, use_agent, nstep, nsampler, nagent)

        output, rnn_hidden_states = self.state_encoder(output, memory.tensor("rnn"), masks)
        return output, rnn_hidden_states
