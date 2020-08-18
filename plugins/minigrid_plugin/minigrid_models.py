import abc
from typing import Callable, Dict, Optional

import gym
import numpy as np
import torch
from gym.spaces.dict import Dict as SpaceDict
from torch import nn

from core.models.basic_models import LinearActorCritic, RNNActorCritic
from core.algorithms.onpolicy_sync.policy import ActorCriticModel
from core.base_abstractions.distributions import CategoricalDistr
from utils.misc_utils import prepare_locals_for_super


class MiniGridSimpleConvBase(ActorCriticModel[CategoricalDistr], abc.ABC):
    actor_critic: ActorCriticModel

    def __init__(
        self,
        action_space: gym.spaces.Discrete,
        observation_space: SpaceDict,
        num_objects: int,
        num_colors: int,
        num_states: int,
        object_embedding_dim: int = 8,
        **kwargs,
    ):
        super().__init__(action_space=action_space, observation_space=observation_space)

        self.num_objects = num_objects
        self.object_embedding_dim = object_embedding_dim

        vis_input_shape = observation_space["minigrid_ego_image"].shape
        agent_view_x, agent_view_y, view_channels = vis_input_shape
        assert agent_view_x == agent_view_y
        self.agent_view = agent_view_x
        self.view_channels = view_channels

        assert (np.array(vis_input_shape[:2]) >= 3).all(), (
            "MiniGridSimpleConvRNN requires" "that the input size be at least 3x3."
        )

        self.num_channels = 0

        if self.num_objects > 0:
            # Object embedding
            self.object_embedding = nn.Embedding(
                num_embeddings=num_objects, embedding_dim=self.object_embedding_dim
            )
            self.object_channel = self.num_channels
            self.num_channels += 1

        self.num_colors = num_colors
        if self.num_colors > 0:
            # Same dimensionality used for colors and states
            self.color_embedding = nn.Embedding(
                num_embeddings=num_colors, embedding_dim=self.object_embedding_dim
            )
            self.color_channel = self.num_channels
            self.num_channels += 1

        self.num_states = num_states
        if self.num_states > 0:
            self.state_embedding = nn.Embedding(
                num_embeddings=num_states, embedding_dim=self.object_embedding_dim
            )
            self.state_channel = self.num_channels
            self.num_channels += 1

        assert self.num_channels == self.view_channels > 0

        self.ac_key = "enc"
        self.observations_for_ac: Dict[str, Optional[torch.Tensor]] = {
            self.ac_key: None
        }

        self.num_agents = 1

    def forward(self, observations, recurrent_hidden_states, prev_actions, masks):
        minigrid_ego_image = observations["minigrid_ego_image"]
        nsteps, nsamplers, nrow, ncol, nchannels = minigrid_ego_image.shape  # assumes
        nagents = recurrent_hidden_states.shape[3]
        minigrid_ego_image = minigrid_ego_image.unsqueeze(2).expand(
            -1, -1, nagents, -1, -1, -1
        )

        # nbatch, nrow, ncol, nchannels = minigrid_ego_image.shape
        assert nrow == ncol == self.agent_view
        # assert nchannels == self.view_channels == 3
        assert nchannels == self.view_channels == self.num_channels
        embed_list = []
        if self.num_objects > 0:
            ego_object_embeds = self.object_embedding(
                minigrid_ego_image[..., self.object_channel].long()
            )
            embed_list.append(ego_object_embeds)
        if self.num_colors > 0:
            ego_color_embeds = self.color_embedding(
                minigrid_ego_image[..., self.color_channel].long()
            )
            embed_list.append(ego_color_embeds)
        if self.num_states > 0:
            ego_state_embeds = self.state_embedding(
                minigrid_ego_image[..., self.state_channel].long()
            )
            embed_list.append(ego_state_embeds)
        ego_embeds = torch.cat(embed_list, dim=-1)

        self.observations_for_ac[self.ac_key] = ego_embeds.view(
            nsteps, nsamplers, nagents, -1
        )

        # noinspection PyCallingNonCallable
        out, rnn_hidden_states = self.actor_critic(
            observations=self.observations_for_ac,
            recurrent_hidden_states=recurrent_hidden_states,
            prev_actions=prev_actions,
            masks=masks,
        )

        self.observations_for_ac[self.ac_key] = None

        return (out, rnn_hidden_states)


class MiniGridSimpleConvRNN(MiniGridSimpleConvBase):
    def __init__(
        self,
        action_space: gym.spaces.Discrete,
        observation_space: SpaceDict,
        num_objects: int,
        num_colors: int,
        num_states: int,
        object_embedding_dim: int = 8,
        hidden_size=512,
        num_layers=1,
        rnn_type="GRU",
        head_type: Callable[
            ..., ActorCriticModel[CategoricalDistr]
        ] = LinearActorCritic,
        **kwargs,
    ):
        super().__init__(**prepare_locals_for_super(locals()))

        self._hidden_size = hidden_size
        agent_view_x, agent_view_y, view_channels = observation_space[
            "minigrid_ego_image"
        ].shape
        self.actor_critic = RNNActorCritic(
            input_uuid=self.ac_key,
            action_space=action_space,
            observation_space=SpaceDict(
                {
                    self.ac_key: gym.spaces.Box(
                        low=np.float32(-1.0),
                        high=np.float32(1.0),
                        shape=(
                            self.object_embedding_dim
                            * agent_view_x
                            * agent_view_y
                            * view_channels,
                        ),
                    )
                }
            ),
            hidden_size=hidden_size,
            num_layers=num_layers,
            rnn_type=rnn_type,
            head_type=head_type,
        )

        self.train()

    @property
    def num_recurrent_layers(self):
        return self.actor_critic.num_recurrent_layers

    @property
    def recurrent_hidden_state_size(self):
        return self._hidden_size


class MiniGridSimpleConv(MiniGridSimpleConvBase):
    def __init__(
        self,
        action_space: gym.spaces.Discrete,
        observation_space: SpaceDict,
        num_objects: int,
        num_colors: int,
        num_states: int,
        object_embedding_dim: int = 8,
        **kwargs,
    ):
        super().__init__(**prepare_locals_for_super(locals()))

        agent_view_x, agent_view_y, view_channels = observation_space[
            "minigrid_ego_image"
        ].shape
        self.actor_critic = LinearActorCritic(
            self.ac_key,
            action_space=action_space,
            observation_space=SpaceDict(
                {
                    self.ac_key: gym.spaces.Box(
                        low=np.float32(-1.0),
                        high=np.float32(1.0),
                        shape=(
                            self.object_embedding_dim
                            * agent_view_x
                            * agent_view_y
                            * view_channels,
                        ),
                    )
                }
            ),
        )

        self.train()

    @property
    def num_recurrent_layers(self):
        return 0

    @property
    def recurrent_hidden_state_size(self):
        return 0
