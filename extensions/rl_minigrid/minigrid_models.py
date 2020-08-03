from typing import Callable, Dict, Optional

import gym
import numpy as np
import torch
from gym.spaces.dict import Dict as SpaceDict
from torch import nn

from models.basic_models import LinearActorCritic, RNNActorCritic
from onpolicy_sync.policy import ActorCriticModel
from rl_base.distributions import CategoricalDistr


class MiniGridSimpleConvRNN(ActorCriticModel[CategoricalDistr]):
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
    ):
        super().__init__(action_space=action_space, observation_space=observation_space)

        self._hidden_size = hidden_size
        self.num_objects = num_objects
        self.object_embedding_dim = object_embedding_dim

        vis_input_shape = observation_space["minigrid_ego_image"].shape
        agent_view_x, agent_view_y, view_channels = vis_input_shape
        assert agent_view_x == agent_view_y
        self.agent_view = agent_view_x
        self.view_channels = view_channels

        assert (np.array(vis_input_shape[:2]) >= 7).all(), (
            "MiniGridSimpleConvRNN requires" "that the input size be at least 7x7."
        )

        # Object embedding
        self.object_embedding = nn.Embedding(
            num_embeddings=num_objects, embedding_dim=self.object_embedding_dim
        )

        # Same dimensionality used for colors and states
        self.color_embedding = nn.Embedding(
            num_embeddings=num_colors, embedding_dim=self.object_embedding_dim
        )
        self.state_embedding = nn.Embedding(
            num_embeddings=num_states, embedding_dim=self.object_embedding_dim
        )

        self.ac_key = "enc"
        self.observations_for_ac: Dict[str, Optional[torch.Tensor]] = {
            self.ac_key: None
        }
        self.rnn_actor_critic = RNNActorCritic(
            input_key=self.ac_key,
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
        return self.rnn_actor_critic.num_recurrent_layers

    @property
    def recurrent_hidden_state_size(self):
        return self._hidden_size

    def forward(self, observations, recurrent_hidden_states, prev_actions, masks):
        minigrid_ego_image = observations["minigrid_ego_image"]
        nbatch, nrow, ncol, nchannels = minigrid_ego_image.shape
        assert nrow == ncol == self.agent_view
        assert nchannels == self.view_channels == 3
        ego_object_embeds = self.object_embedding(minigrid_ego_image[:, :, :, 0].long())
        ego_color_embeds = self.color_embedding(minigrid_ego_image[:, :, :, 1].long())
        ego_state_embeds = self.state_embedding(minigrid_ego_image[:, :, :, 2].long())
        ego_embeds = torch.cat(
            (ego_object_embeds, ego_color_embeds, ego_state_embeds), dim=-1
        )

        self.observations_for_ac[self.ac_key] = ego_embeds.view(nbatch, -1)

        # noinspection PyCallingNonCallable
        out, rnn_hidden_states = self.rnn_actor_critic(
            observations=self.observations_for_ac,
            recurrent_hidden_states=recurrent_hidden_states,
            prev_actions=prev_actions,
            masks=masks,
        )

        self.observations_for_ac[self.ac_key] = None

        return (out, rnn_hidden_states)
