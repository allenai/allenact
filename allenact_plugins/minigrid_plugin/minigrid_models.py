import abc
from typing import Callable, Dict, Optional, Tuple, cast, Union, Any

import gym
import numpy as np
import torch
from gym.spaces.dict import Dict as SpaceDict
from torch import nn

from allenact.algorithms.onpolicy_sync.policy import (
    ActorCriticModel,
    Memory,
    DistributionType,
    ActorCriticOutput,
    ObservationType,
)
from allenact.base_abstractions.distributions import (
    Distr,
    CategoricalDistr,
    ConditionalDistr,
    SequentialDistr,
)
from allenact.embodiedai.models.basic_models import (
    LinearActorCritic,
    RNNActorCritic,
    RNNStateEncoder,
)
from allenact.utils.misc_utils import prepare_locals_for_super


class MiniGridSimpleConvBase(ActorCriticModel[Distr], abc.ABC):
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

    def forward(  # type:ignore
        self,
        observations: ObservationType,
        memory: Memory,
        prev_actions: torch.Tensor,
        masks: torch.FloatTensor,
    ) -> Tuple[ActorCriticOutput[DistributionType], Optional[Memory]]:
        minigrid_ego_image = cast(torch.Tensor, observations["minigrid_ego_image"])
        use_agent = minigrid_ego_image.shape == 6
        nrow, ncol, nchannels = minigrid_ego_image.shape[-3:]
        nsteps, nsamplers, nagents = masks.shape[:3]

        assert nrow == ncol == self.agent_view
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

        if use_agent:
            self.observations_for_ac[self.ac_key] = ego_embeds.view(
                nsteps, nsamplers, nagents, -1
            )
        else:
            self.observations_for_ac[self.ac_key] = ego_embeds.view(
                nsteps, nsamplers * nagents, -1
            )

        # noinspection PyCallingNonCallable
        out, mem_return = self.actor_critic(
            observations=self.observations_for_ac,
            memory=memory,
            prev_actions=prev_actions,
            masks=masks,
        )

        self.observations_for_ac[self.ac_key] = None

        return out, mem_return


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
        self.memory_key = "rnn"

        self.train()

    @property
    def num_recurrent_layers(self):
        return self.actor_critic.num_recurrent_layers

    @property
    def recurrent_hidden_state_size(self):
        return self._hidden_size

    def _recurrent_memory_specification(self):
        return {
            self.memory_key: (
                (
                    ("layer", self.num_recurrent_layers),
                    ("sampler", None),
                    ("hidden", self.recurrent_hidden_state_size),
                ),
                torch.float32,
            )
        }


class ConditionedLinearActorCriticHead(nn.Module):
    def __init__(
        self, input_size: int, master_actions: int = 2, subpolicy_actions: int = 2
    ):
        super().__init__()
        self.input_size = input_size
        self.master_and_critic = nn.Linear(input_size, master_actions + 1)
        self.embed_higher = nn.Embedding(num_embeddings=2, embedding_dim=input_size)
        self.actor = nn.Linear(2 * input_size, subpolicy_actions)

        nn.init.orthogonal_(self.master_and_critic.weight)
        nn.init.constant_(self.master_and_critic.bias, 0)
        nn.init.orthogonal_(self.actor.weight)
        nn.init.constant_(self.actor.bias, 0)

    def lower_policy(self, *args, **kwargs):
        assert "higher" in kwargs
        assert "state_embedding" in kwargs
        emb = self.embed_higher(kwargs["higher"])
        logits = self.actor(torch.cat([emb, kwargs["state_embedding"]], dim=-1))
        return CategoricalDistr(logits=logits)

    def forward(self, x):
        out = self.master_and_critic(x)

        master_logits = out[..., :-1]
        values = out[..., -1:]
        # noinspection PyArgumentList

        cond1 = ConditionalDistr(
            distr_conditioned_on_input_fn_or_instance=CategoricalDistr(
                logits=master_logits
            ),
            action_group_name="higher",
        )
        cond2 = ConditionalDistr(
            distr_conditioned_on_input_fn_or_instance=lambda *args, **kwargs: ConditionedLinearActorCriticHead.lower_policy(
                self, *args, **kwargs
            ),
            action_group_name="lower",
            state_embedding=x,
        )

        return (
            SequentialDistr(cond1, cond2),
            values.view(*values.shape[:2], -1),  # [steps, samplers, flattened]
        )


class ConditionedLinearActorCritic(ActorCriticModel[SequentialDistr]):
    def __init__(
        self,
        input_uuid: str,
        action_space: gym.spaces.Dict,
        observation_space: SpaceDict,
    ):
        super().__init__(action_space=action_space, observation_space=observation_space)

        assert (
            input_uuid in observation_space.spaces
        ), "ConditionedLinearActorCritic expects only a single observational input."
        self.input_uuid = input_uuid

        box_space: gym.spaces.Box = observation_space[self.input_uuid]
        assert isinstance(box_space, gym.spaces.Box), (
            "ConditionedLinearActorCritic requires that"
            "observation space corresponding to the input uuid is a Box space."
        )
        assert len(box_space.shape) == 1
        self.in_dim = box_space.shape[0]
        self.head = ConditionedLinearActorCriticHead(
            input_size=self.in_dim,
            master_actions=action_space["higher"].n,
            subpolicy_actions=action_space["lower"].n,
        )

    # noinspection PyMethodMayBeStatic
    def _recurrent_memory_specification(self):
        return None

    def forward(self, observations, memory, prev_actions, masks):
        dists, values = self.head(observations[self.input_uuid])

        # noinspection PyArgumentList
        return (
            ActorCriticOutput(distributions=dists, values=values, extras={},),
            None,
        )


class ConditionedRNNActorCritic(ActorCriticModel[SequentialDistr]):
    def __init__(
        self,
        input_uuid: str,
        action_space: gym.spaces.Dict,
        observation_space: SpaceDict,
        hidden_size: int = 128,
        num_layers: int = 1,
        rnn_type: str = "GRU",
        head_type: Callable[
            ..., ActorCriticModel[SequentialDistr]
        ] = ConditionedLinearActorCritic,
    ):
        super().__init__(action_space=action_space, observation_space=observation_space)
        self.hidden_size = hidden_size
        self.rnn_type = rnn_type

        assert (
            input_uuid in observation_space.spaces
        ), "LinearActorCritic expects only a single observational input."
        self.input_uuid = input_uuid

        box_space: gym.spaces.Box = observation_space[self.input_uuid]
        assert isinstance(box_space, gym.spaces.Box), (
            "RNNActorCritic requires that"
            "observation space corresponding to the input uuid is a Box space."
        )
        assert len(box_space.shape) == 1
        self.in_dim = box_space.shape[0]

        self.state_encoder = RNNStateEncoder(
            input_size=self.in_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            rnn_type=rnn_type,
            trainable_masked_hidden_state=True,
        )

        self.head_uuid = "{}_{}".format("rnn", input_uuid)

        self.ac_nonrecurrent_head: ActorCriticModel[DirectedGraphicalModel] = head_type(
            input_uuid=self.head_uuid,
            action_space=action_space,
            observation_space=SpaceDict(
                {
                    self.head_uuid: gym.spaces.Box(
                        low=np.float32(0.0), high=np.float32(1.0), shape=(hidden_size,)
                    )
                }
            ),
        )

        self.memory_key = "rnn"

    @property
    def recurrent_hidden_state_size(self) -> int:
        return self.hidden_size

    @property
    def num_recurrent_layers(self) -> int:
        return self.state_encoder.num_recurrent_layers

    def _recurrent_memory_specification(self):
        return {
            self.memory_key: (
                (
                    ("layer", self.num_recurrent_layers),
                    ("sampler", None),
                    ("hidden", self.recurrent_hidden_state_size),
                ),
                torch.float32,
            )
        }

    def forward(  # type:ignore
        self,
        observations: Dict[str, Union[torch.FloatTensor, Dict[str, Any]]],
        memory: Memory,
        prev_actions: torch.Tensor,
        masks: torch.FloatTensor,
    ) -> Tuple[ActorCriticOutput[DistributionType], Optional[Memory]]:
        rnn_out, mem_return = self.state_encoder(
            x=observations[self.input_uuid],
            hidden_states=memory.tensor(self.memory_key),
            masks=masks,
        )

        # noinspection PyCallingNonCallable
        out, _ = self.ac_nonrecurrent_head(
            observations={self.head_uuid: rnn_out},
            memory=None,
            prev_actions=prev_actions,
            masks=masks,
        )

        # noinspection PyArgumentList
        return (
            out,
            memory.set_tensor(self.memory_key, mem_return),
        )


class ConditionedMiniGridSimpleConvRNN(MiniGridSimpleConvBase):
    def __init__(
        self,
        action_space: gym.spaces.Dict,
        observation_space: SpaceDict,
        num_objects: int,
        num_colors: int,
        num_states: int,
        object_embedding_dim: int = 8,
        hidden_size=512,
        num_layers=1,
        rnn_type="GRU",
        head_type: Callable[
            ..., ActorCriticModel[SequentialDistr]
        ] = ConditionedLinearActorCritic,
        **kwargs,
    ):
        super().__init__(**prepare_locals_for_super(locals()))

        self._hidden_size = hidden_size
        agent_view_x, agent_view_y, view_channels = observation_space[
            "minigrid_ego_image"
        ].shape
        self.actor_critic = ConditionedRNNActorCritic(
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
        self.memory_key = "rnn"

        self.train()

    @property
    def num_recurrent_layers(self):
        return self.actor_critic.num_recurrent_layers

    @property
    def recurrent_hidden_state_size(self):
        return self._hidden_size

    def _recurrent_memory_specification(self):
        return {
            self.memory_key: (
                (
                    ("layer", self.num_recurrent_layers),
                    ("sampler", None),
                    ("hidden", self.recurrent_hidden_state_size),
                ),
                torch.float32,
            )
        }


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
        self.memory_key = None

        self.train()

    @property
    def num_recurrent_layers(self):
        return 0

    @property
    def recurrent_hidden_state_size(self):
        return 0

    # noinspection PyMethodMayBeStatic
    def _recurrent_memory_specification(self):
        return None
