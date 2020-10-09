import typing
from typing import Tuple, Dict, Union, Sequence, Optional

import gym
import torch
import torch.nn as nn
from gym.spaces import Dict as SpaceDict

from core.algorithms.onpolicy_sync.policy import (
    ActorCriticModel,
    LinearActorCriticHead,
    DistributionType,
    Memory,
    ObservationType,
)
from core.base_abstractions.distributions import CategoricalDistr
from core.base_abstractions.misc import ActorCriticOutput
from core.models.basic_models import RNNStateEncoder


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

    def adapt_input(self, observations):
        resnet = observations[self.resnet_uuid]

        use_agent = False
        nagent = 1

        if len(resnet.shape) == 6:
            use_agent = True
            nstep, nsampler, nagent = resnet.shape[:3]
        else:
            nstep, nsampler = resnet.shape[:2]

        observations[self.resnet_uuid] = resnet.view(-1, *resnet.shape[-3:])
        observations[self.goal_uuid] = observations[self.goal_uuid].view(-1, 1)

        return observations, use_agent, nstep, nsampler, nagent

    @staticmethod
    def adapt_output(x, use_agent, nstep, nsampler, nagent):
        if use_agent:
            return x.view(nstep, nsampler, nagent, -1)
        return x.view(nstep, nsampler * nagent, -1)

    def forward(self, observations):
        observations, use_agent, nstep, nsampler, nagent = self.adapt_input(
            observations
        )

        if self.blind:
            return self.embed_class(observations[self.goal_uuid])

        embs = [
            self.compress_resnet(observations),
            self.distribute_target(observations),
        ]

        x = self.target_obs_combiner(torch.cat(embs, dim=-3,))
        x = x.view(x.size(0), -1)  # flatten

        return self.adapt_output(x, use_agent, nstep, nsampler, nagent)


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

        self.actor_critic = LinearActorCriticHead(self.hidden_size, action_space.n)

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

    def _recurrent_memory_specification(self):
        return {
            "rnn_hidden": (
                (
                    ("layer", self.state_encoder.num_recurrent_layers),
                    ("sampler", None),
                    ("hidden", self.hidden_size),
                ),
                torch.float32,
            )
        }

    def get_object_type_encoding(
        self, observations: Dict[str, torch.FloatTensor]
    ) -> torch.FloatTensor:
        """Get the object type encoding from input batched observations."""
        return self.goal_visual_encoder.get_object_type_encoding(observations)

    def forward(  # type:ignore
        self,
        observations: ObservationType,
        memory: Memory,
        prev_actions: torch.Tensor,
        masks: torch.FloatTensor,
    ) -> Tuple[ActorCriticOutput[DistributionType], Optional[Memory]]:
        x = self.goal_visual_encoder(observations)

        x, rnn_hidden_states = self.state_encoder(x, memory.tensor("rnn_hidden"), masks)

        dists, vals = self.actor_critic(x)

        return (
            ActorCriticOutput(distributions=dists, values=vals, extras={},),
            memory.set_tensor("rnn_hidden", rnn_hidden_states),
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

    def adapt_input(self, observations):
        boxes = observations[self.detector_uuid]["frcnn_boxes"]
        classes = observations[self.detector_uuid]["frcnn_classes"]

        use_agent = False
        nagent = 1

        if len(boxes.shape) == 6:
            use_agent = True
            nstep, nsampler, nagent = boxes.shape[:3]
        else:
            nstep, nsampler = boxes.shape[:2]

        observations[self.detector_uuid]["frcnn_boxes"] = boxes.view(
            -1, *boxes.shape[-3:]
        )

        observations[self.detector_uuid]["frcnn_classes"] = classes.view(
            -1, *classes.shape[-3:]
        )

        observations[self.goal_uuid] = observations[self.goal_uuid].view(-1, 1)

        return observations, use_agent, nstep, nsampler, nagent

    @staticmethod
    def adapt_output(x, use_agent, nstep, nsampler, nagent):
        if use_agent:
            return x.view(nstep, nsampler, nagent, -1)
        return x.view(nstep, nsampler * nagent, -1)

    def forward(self, observations):
        observations, use_agent, nstep, nsampler, nagent = self.adapt_input(
            observations
        )

        if self.blind:
            return self.embed_class(observations[self.goal_uuid])

        embs = [
            self.compress_resnet(observations),
            self.embed_boxes(observations),
            self.embed_classes(observations),
            self.distribute_target(observations),
        ]

        x = self.target_obs_combiner(torch.cat(embs, dim=-3,))

        x = x.view(x.size(0), -1)  # flatten

        return self.adapt_output(x, use_agent, nstep, nsampler, nagent)


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

        self.actor_critic = LinearActorCriticHead(self.hidden_size, action_space.n)

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

    def _recurrent_memory_specification(self):
        return {
            "rnn_hidden": (
                (
                    ("layer", self.state_encoder.num_recurrent_layers),
                    ("sampler", None),
                    ("hidden", self.hidden_size),
                ),
                torch.float32,
            )
        }

    def get_object_type_encoding(
        self, observations: Dict[str, torch.FloatTensor]
    ) -> torch.FloatTensor:
        """Get the object type encoding from input batched observations."""
        return self.goal_visual_encoder.get_object_type_encoding(observations)

    def forward(  # type:ignore
        self,
        observations: ObservationType,
        memory: Memory,
        prev_actions: torch.Tensor,
        masks: torch.FloatTensor,
    ) -> Tuple[ActorCriticOutput[DistributionType], Optional[Memory]]:
        x = self.goal_visual_encoder(observations)

        x, rnn_hidden_states = self.state_encoder(x, memory.tensor("rnn_hidden"), masks)

        dists, vals = self.actor_critic(x)

        return (
            ActorCriticOutput(distributions=dists, values=vals, extras={},),
            memory.set_tensor("rnn_hidden", rnn_hidden_states),
        )
