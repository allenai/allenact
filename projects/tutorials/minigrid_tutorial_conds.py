from typing import Dict, Optional, List, Any, cast, Callable, Union, Tuple

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gym_minigrid.envs import EmptyRandomEnv5x5
from gym_minigrid.minigrid import MiniGridEnv
from torch.optim.lr_scheduler import LambdaLR

from allenact.algorithms.onpolicy_sync.losses.imitation import Imitation
from allenact.algorithms.onpolicy_sync.losses.ppo import PPO, PPOConfig
from allenact.algorithms.onpolicy_sync.policy import ActorCriticModel, DistributionType
from allenact.base_abstractions.distributions import (
    CategoricalDistr,
    ConditionalDistr,
    SequentialDistr,
)
from allenact.base_abstractions.experiment_config import ExperimentConfig, TaskSampler
from allenact.base_abstractions.misc import ActorCriticOutput, Memory, RLStepResult
from allenact.base_abstractions.sensor import SensorSuite, ExpertActionSensor
from allenact.embodiedai.models.basic_models import RNNStateEncoder
from allenact.utils.experiment_utils import (
    TrainingPipeline,
    Builder,
    PipelineStage,
    LinearDecay,
)
from allenact.utils.misc_utils import prepare_locals_for_super
from allenact_plugins.minigrid_plugin.minigrid_models import MiniGridSimpleConvBase
from allenact_plugins.minigrid_plugin.minigrid_sensors import EgocentricMiniGridSensor
from allenact_plugins.minigrid_plugin.minigrid_tasks import (
    MiniGridTaskSampler,
    MiniGridTask,
)


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
        observation_space: gym.spaces.Dict,
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
        observation_space: gym.spaces.Dict,
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

        self.ac_nonrecurrent_head: ActorCriticModel[SequentialDistr] = head_type(
            input_uuid=self.head_uuid,
            action_space=action_space,
            observation_space=gym.spaces.Dict(
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
        observation_space: gym.spaces.Dict,
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
            observation_space=gym.spaces.Dict(
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


class ConditionedMiniGridTask(MiniGridTask):
    _ACTION_NAMES = ("left", "right", "forward", "pickup")
    _ACTION_IND_TO_MINIGRID_IND = tuple(
        MiniGridEnv.Actions.__members__[name].value for name in _ACTION_NAMES
    )

    @property
    def action_space(self) -> gym.spaces.Dict:
        return gym.spaces.Dict(
            higher=gym.spaces.Discrete(2), lower=gym.spaces.Discrete(2)
        )

    def _step(self, action: Dict[str, int]) -> RLStepResult:
        assert len(action) == 2, "got action={}".format(action)
        minigrid_obs, reward, self._minigrid_done, info = self.env.step(
            action=(
                self._ACTION_IND_TO_MINIGRID_IND[action["lower"] + 2 * action["higher"]]
            )
        )

        # self.env.render()

        return RLStepResult(
            observation=self.get_observations(minigrid_output_obs=minigrid_obs),
            reward=reward,
            done=self.is_done(),
            info=info,
        )

    def query_expert(self, **kwargs) -> Tuple[int, bool]:
        if kwargs["expert_sensor_group_name"] == "higher":
            if self._minigrid_done:
                raise ValueError("Episode is completed, but expert is still queried.")
                # return 0, False
            self.cached_expert = super().query_expert(**kwargs)
            if self.cached_expert[1]:
                return self.cached_expert[0] // 2, True
            else:
                return 0, False
        else:
            assert hasattr(self, "cached_expert")
            if self.cached_expert[1]:
                res = (self.cached_expert[0] % 2, True)
            else:
                res = (0, False)
            del self.cached_expert
            return res


class MiniGridTutorialExperimentConfig(ExperimentConfig):
    @classmethod
    def tag(cls) -> str:
        return "MiniGridTutorial"

    SENSORS = [
        EgocentricMiniGridSensor(agent_view_size=5, view_channels=3),
        ExpertActionSensor(
            action_space=gym.spaces.Dict(
                higher=gym.spaces.Discrete(2), lower=gym.spaces.Discrete(2)
            )
        ),
    ]

    @classmethod
    def create_model(cls, **kwargs) -> nn.Module:
        return ConditionedMiniGridSimpleConvRNN(
            action_space=gym.spaces.Dict(
                higher=gym.spaces.Discrete(2), lower=gym.spaces.Discrete(2)
            ),
            observation_space=SensorSuite(cls.SENSORS).observation_spaces,
            num_objects=cls.SENSORS[0].num_objects,
            num_colors=cls.SENSORS[0].num_colors,
            num_states=cls.SENSORS[0].num_states,
        )

    @classmethod
    def make_sampler_fn(cls, **kwargs) -> TaskSampler:
        return MiniGridTaskSampler(**kwargs)

    def train_task_sampler_args(
        self,
        process_ind: int,
        total_processes: int,
        devices: Optional[List[int]] = None,
        seeds: Optional[List[int]] = None,
        deterministic_cudnn: bool = False,
    ) -> Dict[str, Any]:
        return self._get_sampler_args(process_ind=process_ind, mode="train")

    def valid_task_sampler_args(
        self,
        process_ind: int,
        total_processes: int,
        devices: Optional[List[int]] = None,
        seeds: Optional[List[int]] = None,
        deterministic_cudnn: bool = False,
    ) -> Dict[str, Any]:
        return self._get_sampler_args(process_ind=process_ind, mode="valid")

    def test_task_sampler_args(
        self,
        process_ind: int,
        total_processes: int,
        devices: Optional[List[int]] = None,
        seeds: Optional[List[int]] = None,
        deterministic_cudnn: bool = False,
    ) -> Dict[str, Any]:
        return self._get_sampler_args(process_ind=process_ind, mode="test")

    def _get_sampler_args(self, process_ind: int, mode: str) -> Dict[str, Any]:
        """Generate initialization arguments for train, valid, and test
        TaskSamplers.

        # Parameters
        process_ind : index of the current task sampler
        mode:  one of `train`, `valid`, or `test`
        """
        if mode == "train":
            max_tasks = None  # infinite training tasks
            task_seeds_list = None  # no predefined random seeds for training
            deterministic_sampling = False  # randomly sample tasks in training
        else:
            max_tasks = 20 + 20 * (
                mode == "test"
            )  # 20 tasks for valid, 40 for test (per sampler)

            # one seed for each task to sample:
            # - ensures different seeds for each sampler, and
            # - ensures a deterministic set of sampled tasks.
            task_seeds_list = list(
                range(process_ind * max_tasks, (process_ind + 1) * max_tasks)
            )

            deterministic_sampling = (
                True  # deterministically sample task in validation/testing
            )

        return dict(
            max_tasks=max_tasks,  # see above
            env_class=self.make_env,  # builder for third-party environment (defined below)
            sensors=self.SENSORS,  # sensors used to return observations to the agent
            env_info=dict(),  # parameters for environment builder (none for now)
            task_seeds_list=task_seeds_list,  # see above
            deterministic_sampling=deterministic_sampling,  # see above
            task_class=ConditionedMiniGridTask,
        )

    @staticmethod
    def make_env(*args, **kwargs):
        return EmptyRandomEnv5x5()

    @classmethod
    def machine_params(cls, mode="train", **kwargs) -> Dict[str, Any]:
        return {
            "nprocesses": 128 if mode == "train" else 16,
            "devices": [],
        }

    @classmethod
    def training_pipeline(cls, **kwargs) -> TrainingPipeline:
        ppo_steps = int(150000)
        return TrainingPipeline(
            named_losses=dict(
                imitation_loss=Imitation(
                    cls.SENSORS[1]
                ),  # 0 is Minigrid, 1 is ExpertActionSensor
                ppo_loss=PPO(**PPOConfig, entropy_method_name="conditional_entropy"),
            ),  # type:ignore
            pipeline_stages=[
                PipelineStage(
                    teacher_forcing=LinearDecay(
                        startp=1.0, endp=0.0, steps=ppo_steps // 2,
                    ),
                    loss_names=["imitation_loss", "ppo_loss"],
                    max_stage_steps=ppo_steps,
                )
            ],
            optimizer_builder=Builder(cast(optim.Optimizer, optim.Adam), dict(lr=1e-4)),
            num_mini_batch=4,
            update_repeats=3,
            max_grad_norm=0.5,
            num_steps=16,
            gamma=0.99,
            use_gae=True,
            gae_lambda=0.95,
            advance_scene_rollout_period=None,
            save_interval=10000,
            metric_accumulate_interval=1,
            lr_scheduler_builder=Builder(
                LambdaLR, {"lr_lambda": LinearDecay(steps=ppo_steps)}  # type:ignore
            ),
        )
