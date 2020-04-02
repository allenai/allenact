"""Baseline models for use in the object navigation task.
Object navigation is currently available as a Task in AI2-THOR and
Facebook's Habitat.
"""
from typing import cast, Tuple, Dict

import torch
import torch.nn as nn
import gym
from gym.spaces.dict import Dict as SpaceDict

from models.basic_models import SimpleCNN, RNNStateEncoder
from onpolicy_sync.policy import (
    ActorCriticModel,
    LinearActorCriticHead,
    LinearCriticHead,
    LinearActorHead
)

from rl_base.common import ActorCriticOutput
from rl_base.distributions import CategoricalDistr


class ObjectNavBaselineActorCritic(ActorCriticModel[CategoricalDistr]):
    """Baseline recurrent actor critic model for object-navigation.
    # Attributes
    action_space : The space of actions available to the agent. Currently only discrete
        actions are allowed (so this space will always be of type `gym.spaces.Discrete`).
    observation_space : The observation space expected by the agent. This observation space
        should include (optionally) 'rgb' images and 'depth' images and is required to
        have a component corresponding to the goal `goal_sensor_uuid`.
    goal_sensor_uuid : The uuid of the sensor of the goal object. See `GoalObjectTypeThorSensor`
        as an example of such a sensor.
    hidden_size : The hidden size of the GRU RNN.
    object_type_embedding_dim: The dimensionality of the embedding corresponding to the goal
        object type.
    """

    def __init__(
        self,
        action_space: gym.spaces.Discrete,
        observation_space: SpaceDict,
        goal_sensor_uuid: str,
        hidden_size=512,
        object_type_embedding_dim=8,
        trainable_masked_hidden_state: bool = False,
        num_rnn_layers=1,
        rnn_type='GRU',
    ):
        """Initializer.
        See class documentation for parameter definitions.
        """
        super().__init__(action_space=action_space, observation_space=observation_space)

        self.goal_sensor_uuid = goal_sensor_uuid
        self._n_object_types = self.observation_space.spaces[self.goal_sensor_uuid].n
        self.hidden_size = hidden_size
        self.object_type_embedding_size = object_type_embedding_dim

        self.visual_encoder = SimpleCNN(self.observation_space, hidden_size)

        self.state_encoder = RNNStateEncoder(
            (0 if self.is_blind else self.recurrent_hidden_state_size)
            + object_type_embedding_dim,
            self.recurrent_hidden_state_size,
            trainable_masked_hidden_state=trainable_masked_hidden_state,
            num_layers=num_rnn_layers,
            rnn_type=rnn_type
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
    def recurrent_hidden_state_size(self) -> int:
        """The recurrent hidden state size of the model."""
        return self.hidden_size

    @property
    def is_blind(self) -> bool:
        """True if the model is blind (e.g. neither 'depth' or 'rgb' is an
        input observation type)."""
        return self.visual_encoder.is_blind

    @property
    def num_recurrent_layers(self) -> int:
        """Number of recurrent hidden layers."""
        return self.state_encoder.num_recurrent_layers

    def get_object_type_encoding(
        self, observations: Dict[str, torch.FloatTensor]
    ) -> torch.FloatTensor:
        """Get the object type encoding from input batched observations."""
        return self.object_type_embedding(
            observations[self.goal_sensor_uuid].to(torch.int64)
        )

    def forward(  # type: ignore
        self,
        observations: Dict[str, torch.FloatTensor],
        rnn_hidden_states: torch.FloatTensor,
        prev_actions: torch.LongTensor,
        masks: torch.FloatTensor,
    ) -> Tuple[ActorCriticOutput, torch.FloatTensor]:
        """Processes input batched observations to produce new actor and critic
        values.
        Processes input batched observations (along with prior hidden states, previous actions,
        and masks denoting which recurrent hidden states should be masked) and returns
        an `ActorCriticOutput` object containing the model's policy (distribution over actions)
        and evaluation of the current state (value).
        # Parameters
        observations : Batched input observations.
        rnn_hidden_states : Hidden states from initial timepoints.
        prev_actions : Tensor of previous actions taken.
        masks : Masks applied to hidden states. See `RNNStateEncoder`.
        # Returns
        Tuple of the `ActorCriticOutput` and recurrent hidden state.
        """
        target_encoding = self.get_object_type_encoding(observations)
        x = [target_encoding]

        if not self.is_blind:
            perception_embed = self.visual_encoder(observations)
            x = [perception_embed] + x

        x_cat = torch.cat(x, dim=1)  # type: ignore
        x_out, rnn_hidden_states = self.state_encoder(x_cat, rnn_hidden_states, masks)

        # distributions, values = self.actor_and_critic(x_out)
        return (
            ActorCriticOutput(distributions=self.actor(x_out), values=self.critic(x_out), extras={}),
            rnn_hidden_states,
        )


class ObjectNavResNetActorCritic(ActorCriticModel[CategoricalDistr]):
    """Baseline recurrent actor critic model for object-navigation.
    # Attributes
    action_space : The space of actions available to the agent. Currently only discrete
        actions are allowed (so this space will always be of type `gym.spaces.Discrete`).
    observation_space : The observation space expected by the agent. This observation space
        should include (optionally) 'rgb' images and 'depth' images and is required to
        have a component corresponding to the goal `goal_sensor_uuid`.
    goal_sensor_uuid : The uuid of the sensor of the goal object. See `GoalObjectTypeThorSensor`
        as an example of such a sensor.
    hidden_size : The hidden size of the GRU RNN.
    object_type_embedding_dim: The dimensionality of the embedding corresponding to the goal
        object type.
    """

    def __init__(
        self,
        action_space: gym.spaces.Discrete,
        observation_space: SpaceDict,
        goal_sensor_uuid: str,
        hidden_size=512,
        object_type_embedding_dim=8,
        trainable_masked_hidden_state: bool = False,
        num_rnn_layers=1,
        rnn_type='GRU',
    ):
        """Initializer.
        See class documentation for parameter definitions.
        """
        super().__init__(action_space=action_space, observation_space=observation_space)

        self.goal_sensor_uuid = goal_sensor_uuid
        self._n_object_types = self.observation_space.spaces[self.goal_sensor_uuid].n
        self.hidden_size = hidden_size
        self.object_type_embedding_size = object_type_embedding_dim

        if 'rgb_resnet' in observation_space.spaces and 'depth_resnet' in observation_space.spaces:
            self.visual_encoder = nn.Linear(4096, hidden_size)
        else:
            self.visual_encoder = nn.Linear(2048, hidden_size)

        self.state_encoder = RNNStateEncoder(
            (0 if self.is_blind else self.recurrent_hidden_state_size)
            + object_type_embedding_dim,
            self.recurrent_hidden_state_size,
            trainable_masked_hidden_state=trainable_masked_hidden_state,
            num_layers=num_rnn_layers,
            rnn_type=rnn_type
        )

        self.actor_and_critic = LinearActorCriticHead(
            self.recurrent_hidden_state_size, action_space.n
        )

        self.object_type_embedding = nn.Embedding(
            num_embeddings=self._n_object_types,
            embedding_dim=object_type_embedding_dim,
        )

        self.train()

    @property
    def recurrent_hidden_state_size(self) -> int:
        """The recurrent hidden state size of the model."""
        return self.hidden_size

    @property
    def is_blind(self) -> bool:
        """True if the model is blind (e.g. neither 'depth' or 'rgb' is an
        input observation type)."""
        return False

    @property
    def num_recurrent_layers(self) -> int:
        """Number of recurrent hidden layers."""
        return self.state_encoder.num_recurrent_layers

    def get_object_type_encoding(
        self, observations: Dict[str, torch.FloatTensor]
    ) -> torch.FloatTensor:
        """Get the object type encoding from input batched observations."""
        return self.object_type_embedding(
            observations[self.goal_sensor_uuid].to(torch.int64)
        )

    def forward(  # type: ignore
        self,
        observations: Dict[str, torch.FloatTensor],
        rnn_hidden_states: torch.FloatTensor,
        prev_actions: torch.LongTensor,
        masks: torch.FloatTensor,
    ) -> Tuple[ActorCriticOutput, torch.FloatTensor]:
        """Processes input batched observations to produce new actor and critic
        values.
        Processes input batched observations (along with prior hidden states, previous actions,
        and masks denoting which recurrent hidden states should be masked) and returns
        an `ActorCriticOutput` object containing the model's policy (distribution over actions)
        and evaluation of the current state (value).
        # Parameters
        observations : Batched input observations.
        rnn_hidden_states : Hidden states from initial timepoints.
        prev_actions : Tensor of previous actions taken.
        masks : Masks applied to hidden states. See `RNNStateEncoder`.
        # Returns
        Tuple of the `ActorCriticOutput` and recurrent hidden state.
        """
        target_encoding = self.get_object_type_encoding(observations)
        x = [target_encoding]

        embs = []
        if "rgb_resnet" in observations:
            embs.append(observations["rgb_resnet"].view(-1, observations["rgb_resnet"].shape[-1]))
        if "depth_resnet" in observations:
            embs.append(observations["depth_resnet"].view(-1, observations["depth_resnet"].shape[-1]))
        perception_emb = torch.cat(embs, dim=1)
        x = [self.visual_encoder(perception_emb)] + x

        x_cat = cast(torch.FloatTensor, torch.cat(x, dim=1))  # type: ignore
        x_out, rnn_hidden_states = self.state_encoder(x_cat, rnn_hidden_states, masks)

        distributions, values = self.actor_and_critic(x_out)
        return (
            ActorCriticOutput(distributions=distributions, values=values, extras={}),
            cast(torch.FloatTensor, rnn_hidden_states),
        )


class ObjectNavNavActorCriticTrainResNet50GRU(ActorCriticModel[CategoricalDistr]):
    def __init__(
        self,
        action_space: gym.spaces.Discrete,
        observation_space: SpaceDict,
        goal_sensor_uuid: str,
        hidden_size=512,
        object_type_embedding_dim=8,
        trainable_masked_hidden_state: bool = False,
        num_rnn_layers=1,
        rnn_type='GRU'
    ):
        super().__init__(action_space=action_space, observation_space=observation_space)

        self.goal_sensor_uuid = goal_sensor_uuid
        self._n_object_types = self.observation_space.spaces[self.goal_sensor_uuid].n
        self.recurrent_hidden_state_size = hidden_size
        self.object_type_embedding_size = object_type_embedding_dim

        self.visual_encoder = nn.Sequential(
            nn.Conv2d(2048, 2048, (1, 1)),
            nn.ReLU(),
            nn.Conv2d(2048, 2048, (1, 1)),
            nn.ReLU(),
            nn.Conv2d(2048, 1024, (1, 1)),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, (1, 1)),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, (1, 1)),
            nn.ReLU(),
            nn.Conv2d(1024, 32, (1, 1)),
            nn.ReLU(),
            # nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(2048, 512)
        )

        self.state_encoder = RNNStateEncoder(
            (0 if self.is_blind else self.recurrent_hidden_state_size)
            + object_type_embedding_dim,
            self.recurrent_hidden_state_size,
            trainable_masked_hidden_state=trainable_masked_hidden_state,
            num_layers=num_rnn_layers,
            rnn_type=rnn_type
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
        return False

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    def get_object_type_encoding(
        self, observations: Dict[str, torch.FloatTensor]
    ) -> torch.FloatTensor:
        """Get the object type encoding from input batched observations."""
        return self.object_type_embedding(
            observations[self.goal_sensor_uuid].to(torch.int64)
        )

    def recurrent_hidden_state_size(self):
        return self._hidden_size

    def forward(self, observations, rnn_hidden_states, prev_actions, masks):
        target_encoding = self.get_object_type_encoding(observations)
        x = [target_encoding]

        embs = []
        if "rgb_resnet" in observations:
            embs.append(self.visual_encoder(observations["rgb_resnet"]))
        if "depth_resnet" in observations:
            embs.append(self.visual_encoder(observations["depth_resnet"]))
        perception_emb = torch.cat(embs, dim=1)
        x = [self.visual_encoder(perception_emb)] + x

        x_cat = cast(torch.FloatTensor, torch.cat(x, dim=1))  # type: ignore
        x_out, rnn_hidden_states = self.state_encoder(x_cat, rnn_hidden_states, masks)

        return (
            ActorCriticOutput(
                distributions=self.actor(x_out), values=self.critic(x_out), extras={}
            ),
            rnn_hidden_states,
        )
