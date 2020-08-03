"""Baseline models for use in the object navigation task.
Object navigation is currently available as a Task in AI2-THOR and
Facebook's Habitat.
"""
from typing import cast, Tuple, Dict, Optional
import typing

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
        self._hidden_size = hidden_size
        self.object_type_embedding_size = object_type_embedding_dim

        self.visual_encoder = SimpleCNN(self.observation_space, self._hidden_size)

        self.state_encoder = RNNStateEncoder(
            (0 if self.is_blind else self._hidden_size)
            + object_type_embedding_dim,
            self._hidden_size,
            trainable_masked_hidden_state=trainable_masked_hidden_state,
            num_layers=num_rnn_layers,
            rnn_type=rnn_type
        )

        self.actor = LinearActorHead(
            self._hidden_size, action_space.n
        )
        self.critic = LinearCriticHead(
            self._hidden_size
        )

        self.object_type_embedding = nn.Embedding(
            num_embeddings=self._n_object_types,
            embedding_dim=object_type_embedding_dim,
        )

        self.train()

    @property
    def recurrent_hidden_state_size(self) -> int:
        """The recurrent hidden state size of the model."""
        return self._hidden_size

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
        self._hidden_size = hidden_size
        self.object_type_embedding_size = object_type_embedding_dim

        if 'rgb_resnet' in observation_space.spaces and 'depth_resnet' in observation_space.spaces:
            self.visual_encoder = nn.Linear(4096, hidden_size)
        else:
            self.visual_encoder = nn.Linear(2048, hidden_size)

        self.state_encoder = RNNStateEncoder(
            (0 if self.is_blind else self._hidden_size)
            + object_type_embedding_dim,
            self._hidden_size,
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
        return self._hidden_size

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


class ObjectNavActorCriticTrainResNet50RNN(ActorCriticModel[CategoricalDistr]):
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
        self._hidden_size = hidden_size
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
            (0 if self.is_blind else self._hidden_size)
            + object_type_embedding_dim,
            self._hidden_size,
            trainable_masked_hidden_state=trainable_masked_hidden_state,
            num_layers=num_rnn_layers,
            rnn_type=rnn_type
        )

        self.actor = LinearActorHead(
            self._hidden_size, action_space.n
        )
        self.critic = LinearCriticHead(
            self._hidden_size
        )

        self.object_type_embedding = nn.Embedding(
            num_embeddings=self._n_object_types,
            embedding_dim=object_type_embedding_dim,
        )

        self.train()

    @property
    def output_size(self):
        return self._hidden_size

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
        x = [perception_emb] + x

        x_cat = cast(torch.FloatTensor, torch.cat(x, dim=1))  # type: ignore
        x_out, rnn_hidden_states = self.state_encoder(x_cat, rnn_hidden_states, masks)

        return (
            ActorCriticOutput(
                distributions=self.actor(x_out), values=self.critic(x_out), extras={}
            ),
            rnn_hidden_states,
        )


class ResnetTensorObjectNavActorCritic(ActorCriticModel[CategoricalDistr]):
    def __init__(
            self,
            action_space: gym.spaces.Discrete,
            observation_space: SpaceDict,
            goal_sensor_uuid: str,
            rgb_resnet_preprocessor_uuid: Optional[str],
            depth_resnet_preprocessor_uuid: Optional[str] = None,
            hidden_size: int = 512,
            goal_dims: int = 32,
            resnet_compressor_hidden_out_dims: Tuple[int, int] = (128, 32),
            combiner_hidden_out_dims: Tuple[int, int] = (128, 32),
    ):

        super().__init__(
            action_space=action_space, observation_space=observation_space,
        )

        self._hidden_size = hidden_size
        if rgb_resnet_preprocessor_uuid is None or depth_resnet_preprocessor_uuid is None:
            resnet_preprocessor_uuid = rgb_resnet_preprocessor_uuid if rgb_resnet_preprocessor_uuid is None else \
                depth_resnet_preprocessor_uuid
            self.goal_visual_encoder = ResnetTensorGoalEncoder(
                self.observation_space,
                goal_sensor_uuid,
                resnet_preprocessor_uuid,
                goal_dims,
                resnet_compressor_hidden_out_dims,
                combiner_hidden_out_dims,
            )
        else:
            self.goal_visual_encoder = ResnetDualTensorGoalEncoder(
                self.observation_space,
                goal_sensor_uuid,
                rgb_resnet_preprocessor_uuid,
                depth_resnet_preprocessor_uuid,
                goal_dims,
                resnet_compressor_hidden_out_dims,
                combiner_hidden_out_dims,
            )
        self.state_encoder = RNNStateEncoder(
            self.goal_visual_encoder.output_dims, self._hidden_size,
        )
        self.actor = LinearActorHead(self._hidden_size, action_space.n)
        self.critic = LinearCriticHead(self._hidden_size)
        self.train()

    @property
    def recurrent_hidden_state_size(self) -> int:
        """The recurrent hidden state size of the model."""
        return self._hidden_size

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


class ResNextTrainObjectNavActorCritic(ActorCriticModel[CategoricalDistr]):
    def __init__(
            self,
            action_space: gym.spaces.Discrete,
            observation_space: SpaceDict,
            goal_sensor_uuid: str,
            resnet_preprocessor_uuid: str,
            hidden_size: int = 512,
            goal_dims: int = 32,
            resnet_compressor_hidden_out_dims: Tuple[int, int] = (128, 32),
            combiner_hidden_out_dims: Tuple[int, int] = (128, 32),
    ):

        super().__init__(
            action_space=action_space, observation_space=observation_space,
        )

        self._hidden_size = hidden_size
        self.goal_visual_encoder = PointNavResNetPolicy(
            observation_space=observation_space,
            action_space=action_space,
            goal_sensor_uuid=goal_sensor_uuid,
            backbone=visual_encoder_weights['model_args'].backbone,
            hidden_size=visual_encoder_weights['model_args'].hidden_size,
            num_recurrent_layers=visual_encoder_weights['model_args'].num_recurrent_layers,
            resnet_baseplanes=visual_encoder_weights['model_args'].resnet_baseplanes,
            rnn_type=visual_encoder_weights['model_args'].rnn_type
        )

        self.state_encoder = RNNStateEncoder(
            self.goal_visual_encoder.output_dims, self._hidden_size,
        )
        self.actor = LinearActorHead(self._hidden_size, action_space.n)
        self.critic = LinearCriticHead(self._hidden_size)
        self.train()

    @property
    def recurrent_hidden_state_size(self) -> int:
        """The recurrent hidden state size of the model."""
        return self._hidden_size

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
        return target_emb.view(-1, self.class_dims, 1, 1).expand(-1, -1, self.resnet_tensor_shape[-2],
                                                                 self.resnet_tensor_shape[-1])

    def forward(self, observations):
        if self.blind:
            return self.embed_class(observations[self.goal_uuid])
        embs = [
            self.compress_resnet(observations),
            self.distribute_target(observations),
        ]
        x = self.target_obs_combiner(torch.cat(embs, dim=1, ))
        return x.view(x.size(0), -1)  # flatten


class ResnetDualTensorGoalEncoder(nn.Module):
    def __init__(
            self,
            observation_spaces: SpaceDict,
            goal_sensor_uuid: str,
            rgb_resnet_preprocessor_uuid: str,
            depth_resnet_preprocessor_uuid: str,
            class_dims: int = 32,
            resnet_compressor_hidden_out_dims: Tuple[int, int] = (128, 32),
            combiner_hidden_out_dims: Tuple[int, int] = (128, 32),
    ) -> None:
        super().__init__()
        self.goal_uuid = goal_sensor_uuid
        self.rgb_resnet_uuid = rgb_resnet_preprocessor_uuid
        self.depth_resnet_uuid = depth_resnet_preprocessor_uuid
        self.class_dims = class_dims
        self.resnet_hid_out_dims = resnet_compressor_hidden_out_dims
        self.combine_hid_out_dims = combiner_hidden_out_dims
        self.embed_class = nn.Embedding(
            num_embeddings=observation_spaces.spaces[self.goal_uuid].n,
            embedding_dim=self.class_dims,
        )
        self.blind = self.rgb_resnet_uuid not in observation_spaces.spaces or \
                     self.depth_resnet_uuid not in observation_spaces.spaces
        if not self.blind:
            self.resnet_tensor_shape = observation_spaces.spaces[self.rgb_resnet_uuid].shape
            self.rgb_resnet_compressor = nn.Sequential(
                nn.Conv2d(self.resnet_tensor_shape[0], self.resnet_hid_out_dims[0], 1),
                nn.ReLU(),
                nn.Conv2d(*self.resnet_hid_out_dims[0:2], 1),
                nn.ReLU(),
            )
            self.depth_resnet_compressor = nn.Sequential(
                nn.Conv2d(self.resnet_tensor_shape[0], self.resnet_hid_out_dims[0], 1),
                nn.ReLU(),
                nn.Conv2d(*self.resnet_hid_out_dims[0:2], 1),
                nn.ReLU(),
            )
            self.rgb_target_obs_combiner = nn.Sequential(
                nn.Conv2d(
                    self.resnet_hid_out_dims[1] + self.class_dims,
                    self.combine_hid_out_dims[0],
                    1,
                ),
                nn.ReLU(),
                nn.Conv2d(*self.combine_hid_out_dims[0:2], 1),
            )
            self.depth_target_obs_combiner = nn.Sequential(
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
                    2
                    * self.combine_hid_out_dims[-1]
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

    def compress_rgb_resnet(self, observations):
        return self.rgb_resnet_compressor(observations[self.rgb_resnet_uuid])

    def compress_depth_resnet(self, observations):
        return self.depth_resnet_compressor(observations[self.depth_resnet_uuid])

    def distribute_target(self, observations):
        target_emb = self.embed_class(observations[self.goal_uuid])
        return target_emb.view(-1, self.class_dims, 1, 1).expand(-1, -1, self.resnet_tensor_shape[-2],
                                                                 self.resnet_tensor_shape[-1])

    def forward(self, observations):
        if self.blind:
            return self.embed_class(observations[self.goal_uuid])
        rgb_embs = [
            self.compress_rgb_resnet(observations),
            self.distribute_target(observations),
        ]
        rgb_x = self.rgb_target_obs_combiner(torch.cat(rgb_embs, dim=1, ))
        depth_embs = [
            self.compress_depth_resnet(observations),
            self.distribute_target(observations),
        ]
        depth_x = self.depth_target_obs_combiner(torch.cat(depth_embs, dim=1, ))
        x = torch.cat([rgb_x, depth_x], dim=1)
        return x.view(x.size(0), -1)  # flatten
