import typing
from typing import Tuple, Dict, Union, Sequence, Optional

import gym
import torch
import torch.nn as nn
from gym.spaces import Dict as SpaceDict

from core.models.basic_models import RNNStateEncoder
from core.algorithms.onpolicy_sync.policy import (
    ActorCriticModel,
    LinearActorHead,
    LinearCriticHead,
)
from core.base_abstractions.misc import ActorCriticOutput
from core.base_abstractions.distributions import CategoricalDistr


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
    def recurrent_hidden_state_size(
        self,
    ) -> Union[int, Dict[str, Tuple[Sequence[Tuple[str, Optional[int]]], torch.dtype]]]:
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


class ResnetTensorObjectNavActorCriticMemory(ResnetTensorObjectNavActorCritic):
    @property
    def recurrent_hidden_state_size(
        self,
    ) -> Dict[str, Tuple[Sequence[Tuple[str, Optional[int]]], torch.dtype]]:
        """The memory spec of the model: A dictionary with string keys (memory type identification) and
        tuple values (memory type specification), each with:
        1. memory shape, e.g. `(("step", None), ("layer", 2), ("sampler", None), ("agent", 1), ("hidden", 32))`
        for an RNN, where the `step` dimension placeholder is *always* the first dimension,
        the `sampler` dimension placeholder *always* precedes the `agent` dimension,
        the `agent` dimension has the number of agents in the model and it's *always* the next one after `sampler`,
        and `layer` and `hidden` correspond to the standard RNN hidden state parametrization;
        2. the data type, e.g. `torch.float32`."""
        return {
            "rnn_hidden": (
                (
                    ("step", None),
                    ("layer", self.state_encoder.num_recurrent_layers),
                    ("sampler", None),
                    ("agent", 1),
                    ("hidden", self.hidden_size),
                ),
                torch.float32,
            )
        }

    @property
    def num_recurrent_layers(self) -> int:
        """Returns -1, indicating we are using a memory specification in
        recurrent_hidden_state_size."""
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


class ResnetFasterRCNNTensorsGoalEncoder(nn.Module):
    def __init__(
        self,
        observation_spaces: SpaceDict,
        goal_sensor_uuid: str,
        resnet_preprocessor_uuid: str,
        detector_preprocessor_uuid: str,
        class_dims: int = 32,
        max_dets: int = 3,
        resnet_compressor_hidden_out_dims: Tuple[int, int] = (128, 32),
        box_embedder_hidden_out_dims: Tuple[int, int] = (128, 32),
        class_embedder_hidden_out_dims: Tuple[int, int] = (128, 32),
        combiner_hidden_out_dims: Tuple[int, int] = (128, 32),
    ) -> None:
        super().__init__()

        self.goal_uuid = goal_sensor_uuid
        self.resnet_uuid = resnet_preprocessor_uuid
        self.detector_uuid = detector_preprocessor_uuid

        self.class_dims = class_dims
        self.max_dets = max_dets

        self.resnet_hid_out_dims = resnet_compressor_hidden_out_dims
        self.box_hid_out_dims = box_embedder_hidden_out_dims
        self.class_hid_out_dims = class_embedder_hidden_out_dims
        self.combine_hid_out_dims = combiner_hidden_out_dims

        self.embed_class = nn.Embedding(
            num_embeddings=observation_spaces.spaces[self.goal_uuid].n,
            embedding_dim=self.class_dims,
        )

        self.blind = (self.resnet_uuid not in observation_spaces.spaces) and (
            self.detector_uuid not in observation_spaces.spaces
        )

        if not self.blind:
            self.resnet_tensor_shape = observation_spaces.spaces[self.resnet_uuid].shape

            self.resnet_compressor = nn.Sequential(
                nn.Conv2d(self.resnet_tensor_shape[0], self.resnet_hid_out_dims[0], 1),
                nn.ReLU(),
                nn.Conv2d(*self.resnet_hid_out_dims[0:2], 1),
                nn.ReLU(),
            )

            self.box_tensor_shape = (
                observation_spaces.spaces[self.detector_uuid]
                .spaces["frcnn_boxes"]
                .shape
            )
            assert (
                self.box_tensor_shape[1:] == self.resnet_tensor_shape[1:]
            ), "Spatial dimensions of object detector and resnet tensor do not match: {} vs {}".format(
                self.box_tensor_shape, self.resnet_tensor_shape
            )

            self.box_embedder = nn.Sequential(
                nn.Conv2d(self.box_tensor_shape[0], self.box_hid_out_dims[0], 1),
                nn.ReLU(),
                nn.Conv2d(*self.box_hid_out_dims[0:2], 1),
                nn.ReLU(),
            )

            self.class_combiner = nn.Sequential(
                nn.Conv2d(
                    self.max_dets * self.class_dims, self.class_hid_out_dims[0], 1
                ),
                nn.ReLU(),
                nn.Conv2d(*self.class_hid_out_dims[0:2], 1),
                nn.ReLU(),
            )

            self.target_obs_combiner = nn.Sequential(
                nn.Conv2d(
                    self.resnet_hid_out_dims[1]
                    + self.box_hid_out_dims[1]
                    + self.class_hid_out_dims[1]
                    + self.class_dims,
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

    def embed_classes(self, observations):
        classes = observations[self.detector_uuid]["frcnn_classes"]
        classes = classes.permute(0, 2, 3, 1).contiguous()  # move classes to last dim
        classes_shape = classes.shape
        class_emb = self.embed_class(classes.view(-1))  # (flattened)
        class_emb = class_emb.view(
            classes_shape[:-1] + (self.max_dets * class_emb.shape[-1],)
        )  # align embedding along last dimension
        class_emb = class_emb.permute(
            0, 3, 1, 2
        ).contiguous()  # convert into image tensor
        return self.class_combiner(class_emb)

    def embed_boxes(self, observations):
        return self.box_embedder(observations[self.detector_uuid]["frcnn_boxes"])

    def forward(self, observations):
        if self.blind:
            return self.embed_class(observations[self.goal_uuid])

        embs = [
            self.compress_resnet(observations),
            self.embed_boxes(observations),
            self.embed_classes(observations),
            self.distribute_target(observations),
        ]

        x = self.target_obs_combiner(torch.cat(embs, dim=-3,))

        return x.view(x.size(0), -1)  # flatten


class ResnetFasterRCNNTensorsObjectNavActorCritic(ActorCriticModel[CategoricalDistr]):
    def __init__(
        self,
        action_space: gym.spaces.Discrete,
        observation_space: SpaceDict,
        goal_sensor_uuid: str,
        resnet_preprocessor_uuid: str,
        detector_preprocessor_uuid: str,
        rnn_hidden_size=512,
        goal_dims: int = 32,
        max_dets: int = 3,
        resnet_compressor_hidden_out_dims: Tuple[int, int] = (128, 32),
        box_embedder_hidden_out_dims: Tuple[int, int] = (128, 32),
        class_embedder_hidden_out_dims: Tuple[int, int] = (128, 32),
        combiner_hidden_out_dims: Tuple[int, int] = (128, 32),
    ):
        super().__init__(
            action_space=action_space, observation_space=observation_space,
        )

        self.hidden_size = rnn_hidden_size

        self.goal_visual_encoder = ResnetFasterRCNNTensorsGoalEncoder(
            self.observation_space,
            goal_sensor_uuid,
            resnet_preprocessor_uuid,
            detector_preprocessor_uuid,
            goal_dims,
            max_dets,
            resnet_compressor_hidden_out_dims,
            box_embedder_hidden_out_dims,
            class_embedder_hidden_out_dims,
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
