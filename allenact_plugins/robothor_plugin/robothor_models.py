from typing import Tuple, Optional

import gym
import torch
from gym.spaces import Dict as SpaceDict

from allenact.algorithms.onpolicy_sync.policy import (
    ActorCriticModel,
    LinearActorCriticHead,
    DistributionType,
    Memory,
    ObservationType,
)
from allenact.base_abstractions.misc import ActorCriticOutput
from allenact.embodiedai.models.basic_models import RNNStateEncoder, SimpleCNN
from allenact_plugins.robothor_plugin.robothor_distributions import (
    TupleCategoricalDistr,
)


class TupleLinearActorCriticHead(LinearActorCriticHead):
    def forward(self, x):
        out = self.actor_and_critic(x)

        logits = out[..., :-1]
        values = out[..., -1:]
        # noinspection PyArgumentList
        return (
            TupleCategoricalDistr(logits=logits),  # [steps, samplers, ...]
            values.view(*values.shape[:2], -1),  # [steps, samplers, flattened]
        )


class NavToPartnerActorCriticSimpleConvRNN(ActorCriticModel[TupleCategoricalDistr]):
    action_space: gym.spaces.Tuple

    def __init__(
        self,
        action_space: gym.spaces.Tuple,
        observation_space: SpaceDict,
        rgb_uuid: Optional[str] = "rgb",
        hidden_size=512,
        num_rnn_layers=1,
        rnn_type="GRU",
    ):
        super().__init__(action_space=action_space, observation_space=observation_space)

        self._hidden_size = hidden_size

        self.rgb_uuid = rgb_uuid

        self.visual_encoder = SimpleCNN(
            observation_space=observation_space,
            output_size=hidden_size,
            rgb_uuid=self.rgb_uuid,
            depth_uuid=None,
        )

        self.state_encoder = RNNStateEncoder(
            0 if self.is_blind else self.recurrent_hidden_state_size,
            self._hidden_size,
            num_layers=num_rnn_layers,
            rnn_type=rnn_type,
        )

        self.actor_critic = TupleLinearActorCriticHead(
            self._hidden_size, action_space[0].n
        )

        self.train()

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def is_blind(self):
        return self.visual_encoder.is_blind

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    @property
    def recurrent_hidden_state_size(self):
        return self._hidden_size

    @property
    def num_agents(self):
        return len(self.action_space)

    def _recurrent_memory_specification(self):
        return dict(
            rnn=(
                (
                    ("layer", self.num_recurrent_layers),
                    ("sampler", None),
                    ("agent", self.num_agents),
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
        if not self.is_blind:
            perception_embed = self.visual_encoder(observations)
        else:
            # TODO manage blindness for all agents simultaneously or separate?
            raise NotImplementedError()

        # TODO alternative where all agents consume all observations
        x, rnn_hidden_states = self.state_encoder(
            perception_embed, memory.tensor("rnn"), masks
        )

        dists, vals = self.actor_critic(x)

        return (
            ActorCriticOutput(distributions=dists, values=vals, extras={},),
            memory.set_tensor("rnn", rnn_hidden_states),
        )
