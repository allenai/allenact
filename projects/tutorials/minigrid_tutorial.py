from typing import Dict, Optional, List, Any

import gym
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import LambdaLR

from rl_base.experiment_config import ExperimentConfig, TaskSampler
from utils.experiment_utils import TrainingPipeline, Builder, PipelineStage, LinearDecay
from extensions.rl_minigrid.minigrid_tasks import MiniGridTaskSampler, MiniGridTask
from extensions.rl_minigrid.minigrid_sensors import EgocentricMiniGridSensor
from gym_minigrid.envs import EmptyRandomEnv5x5
from extensions.rl_minigrid.minigrid_models import MiniGridSimpleConvRNN
from rl_base.sensor import SensorSuite
from onpolicy_sync.losses.ppo import PPO, PPOConfig


class MiniGridTutorialExperimentConfig(ExperimentConfig):
    @classmethod
    def tag(cls) -> str:
        return "MiniGridTutorial"

    SENSORS = [
        EgocentricMiniGridSensor(agent_view_size=10, view_channels=3),
    ]

    @classmethod
    def training_pipeline(cls, **kwargs) -> TrainingPipeline:
        ppo_steps = int(150000)
        return TrainingPipeline(
            named_losses=dict(ppo_loss=Builder(PPO, kwargs={}, default=PPOConfig,)),
            pipeline_stages=[
                PipelineStage(loss_names=["ppo_loss"], max_stage_steps=ppo_steps)
            ],
            optimizer_builder=Builder(optim.Adam, dict(lr=1e-4)),
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
                LambdaLR, {"lr_lambda": LinearDecay(steps=ppo_steps)}
            ),
        )

    @classmethod
    def machine_params(cls, mode="train", **kwargs) -> Dict[str, Any]:
        return {
            "nprocesses": 128 if mode == "train" else 16,
            "gpu_ids": [],
        }

    @classmethod
    def create_model(cls, **kwargs) -> nn.Module:
        return MiniGridSimpleConvRNN(
            action_space=gym.spaces.Discrete(len(MiniGridTask.class_action_names())),
            observation_space=SensorSuite(cls.SENSORS).observation_spaces,
            num_objects=cls.SENSORS[0].num_objects,
            num_colors=cls.SENSORS[0].num_colors,
            num_states=cls.SENSORS[0].num_states,
        )

    @staticmethod
    def make_env(*args, **kwargs):
        return EmptyRandomEnv5x5()

    def _get_sampler_args(self, process_ind: int, mode: str) -> Dict[str, Any]:
        num_eval_tasks_per_sampler = 20 if mode == "valid" else 40
        return dict(
            env_class=self.make_env,
            sensors=self.SENSORS,
            env_info=dict(),
            max_tasks=None if mode == "train" else num_eval_tasks_per_sampler,
            deterministic_sampling=False if mode == "train" else True,
            task_seeds_list=None
            if mode == "train"
            else list(
                range(
                    process_ind * num_eval_tasks_per_sampler,
                    (process_ind + 1) * num_eval_tasks_per_sampler,
                )
            ),
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
        """Specifies the validation parameters for the `process_ind`th
        validation process.

        See `ExperimentConfig.train_task_sampler_args` for parameter
        definitions.
        """
        return self._get_sampler_args(process_ind=process_ind, mode="valid")

    def test_task_sampler_args(
        self,
        process_ind: int,
        total_processes: int,
        devices: Optional[List[int]] = None,
        seeds: Optional[List[int]] = None,
        deterministic_cudnn: bool = False,
    ) -> Dict[str, Any]:
        """Specifies the test parameters for the `process_ind`th test process.

        See `ExperimentConfig.train_task_sampler_args` for parameter
        definitions.
        """
        return self._get_sampler_args(process_ind=process_ind, mode="test")
