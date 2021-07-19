"""Baseline models for use in the Arm Point Navigation task.

Arm Point Navigation is currently available as a Task in ManipulaTHOR.
"""
from typing import Tuple, Optional

import gym
import torch
from allenact.algorithms.onpolicy_sync.policy import (
    ActorCriticModel,
    LinearCriticHead,
    DistributionType,
    Memory,
    ObservationType,
)
from allenact.base_abstractions.distributions import CategoricalDistr
from allenact.base_abstractions.misc import ActorCriticOutput
from allenact.embodiedai.models.basic_models import SimpleCNN, RNNStateEncoder
from gym.spaces.dict import Dict as SpaceDict

from projects.manipulathor_baselines.armpointnav_baselines.models.base_models import (
    LinearActorHeadNoCategory,
)
from projects.manipulathor_baselines.armpointnav_baselines.models.manipulathor_net_utils import (
    input_embedding_net,
)


class DisjointArmPointNavBaselineActorCritic(ActorCriticModel[CategoricalDistr]):
    """Disjoint Baseline recurrent actor critic model for armpointnav.

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
        hidden_size=512,
        obj_state_embedding_size=512,
        trainable_masked_hidden_state: bool = False,
        num_rnn_layers=1,
        rnn_type="GRU",
    ):
        """Initializer.

        See class documentation for parameter definitions.
        """
        super().__init__(action_space=action_space, observation_space=observation_space)

        self._hidden_size = hidden_size
        self.object_type_embedding_size = obj_state_embedding_size

        self.visual_encoder_pick = SimpleCNN(
            self.observation_space,
            self._hidden_size,
            rgb_uuid=None,
            depth_uuid="depth_lowres",
        )
        self.visual_encoder_drop = SimpleCNN(
            self.observation_space,
            self._hidden_size,
            rgb_uuid=None,
            depth_uuid="depth_lowres",
        )

        self.state_encoder = RNNStateEncoder(
            (self._hidden_size) + obj_state_embedding_size,
            self._hidden_size,
            trainable_masked_hidden_state=trainable_masked_hidden_state,
            num_layers=num_rnn_layers,
            rnn_type=rnn_type,
        )

        self.actor_pick = LinearActorHeadNoCategory(self._hidden_size, action_space.n)
        self.critic_pick = LinearCriticHead(self._hidden_size)
        self.actor_drop = LinearActorHeadNoCategory(self._hidden_size, action_space.n)
        self.critic_drop = LinearCriticHead(self._hidden_size)

        # self.object_state_embedding = nn.Embedding(num_embeddings=6, embedding_dim=obj_state_embedding_size)

        relative_dist_embedding_size = torch.Tensor([3, 100, obj_state_embedding_size])
        self.relative_dist_embedding_pick = input_embedding_net(
            relative_dist_embedding_size.long().tolist(), dropout=0
        )
        self.relative_dist_embedding_drop = input_embedding_net(
            relative_dist_embedding_size.long().tolist(), dropout=0
        )

        self.train()

    @property
    def recurrent_hidden_state_size(self) -> int:
        """The recurrent hidden state size of the model."""
        return self._hidden_size

    @property
    def num_recurrent_layers(self) -> int:
        """Number of recurrent hidden layers."""
        return self.state_encoder.num_recurrent_layers

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

    def forward(  # type:ignore
        self,
        observations: ObservationType,
        memory: Memory,
        prev_actions: torch.Tensor,
        masks: torch.FloatTensor,
    ) -> Tuple[ActorCriticOutput[DistributionType], Optional[Memory]]:
        """Processes input batched observations to produce new actor and critic
        values. Processes input batched observations (along with prior hidden
        states, previous actions, and masks denoting which recurrent hidden
        states should be masked) and returns an `ActorCriticOutput` object
        containing the model's policy (distribution over actions) and
        evaluation of the current state (value).

        # Parameters
        observations : Batched input observations.
        memory : `Memory` containing the hidden states from initial timepoints.
        prev_actions : Tensor of previous actions taken.
        masks : Masks applied to hidden states. See `RNNStateEncoder`.
        # Returns
        Tuple of the `ActorCriticOutput` and recurrent hidden state.
        """

        arm2obj_dist = self.relative_dist_embedding_pick(
            observations["relative_agent_arm_to_obj"]
        )
        obj2goal_dist = self.relative_dist_embedding_drop(
            observations["relative_obj_to_goal"]
        )

        perception_embed_pick = self.visual_encoder_pick(observations)
        perception_embed_drop = self.visual_encoder_drop(observations)

        pickup_bool = observations["pickedup_object"]
        after_pickup = pickup_bool == 1
        distances = arm2obj_dist
        distances[after_pickup] = obj2goal_dist[after_pickup]

        perception_embed = perception_embed_pick
        perception_embed[after_pickup] = perception_embed_drop[after_pickup]

        x = [distances, perception_embed]

        x_cat = torch.cat(x, dim=-1)  # type: ignore
        x_out, rnn_hidden_states = self.state_encoder(
            x_cat, memory.tensor("rnn"), masks
        )
        actor_out_pick = self.actor_pick(x_out)
        critic_out_pick = self.critic_pick(x_out)

        actor_out_drop = self.actor_drop(x_out)
        critic_out_drop = self.critic_drop(x_out)

        actor_out = actor_out_pick
        actor_out[after_pickup] = actor_out_drop[after_pickup]
        critic_out = critic_out_pick
        critic_out[after_pickup] = critic_out_drop[after_pickup]

        actor_out = CategoricalDistr(logits=actor_out)
        actor_critic_output = ActorCriticOutput(
            distributions=actor_out, values=critic_out, extras={}
        )
        updated_memory = memory.set_tensor("rnn", rnn_hidden_states)

        return (
            actor_critic_output,
            updated_memory,
        )
