from typing import Dict, Optional, List, Any

import gym
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import LambdaLR

from core.algorithms.onpolicy_sync.losses.ppo import PPO
from core.base_abstractions.experiment_config import ExperimentConfig, TaskSampler
from core.base_abstractions.sensor import SensorSuite
from plugins.gym_plugin.gym_models import MemorylessActorCritic
from plugins.gym_plugin.gym_sensors import GymBox2DSensor
from plugins.gym_plugin.gym_tasks import GymTaskSampler
from utils.experiment_utils import TrainingPipeline, Builder, PipelineStage, LinearDecay


class GymTutorialExperimentConfig(ExperimentConfig):
    """An experiment is identified by a `tag`."""

    @classmethod
    def tag(cls) -> str:
        return "GymTutorial"

    SENSORS = [
        GymBox2DSensor("LunarLanderContinuous-v2", uuid="gym_box_data"),
    ]

    @classmethod
    def create_model(cls, **kwargs) -> nn.Module:
        return MemorylessActorCritic(
            input_uuid="gym_box_data",
            action_space=gym.spaces.Box(-1.0, 1.0, (2,)),
            observation_space=SensorSuite(cls.SENSORS).observation_spaces,
            action_std=0.5,
        )

    @classmethod
    def make_sampler_fn(cls, **kwargs) -> TaskSampler:
        return GymTaskSampler(**kwargs)

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
            max_tasks = 3

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
            gym_env_types=["LunarLanderContinuous-v2"],
            sensors=self.SENSORS,  # sensors used to return observations to the agent
            max_tasks=max_tasks,  # see above
            task_seeds_list=task_seeds_list,  # see above
            deterministic_sampling=deterministic_sampling,  # see above
        )

    @classmethod
    def machine_params(cls, mode="train", **kwargs) -> Dict[str, Any]:
        return {
            "nprocesses": 8 if mode == "train" else 1,
            "devices": [],
        }

    @classmethod
    def training_pipeline(cls, **kwargs) -> TrainingPipeline:
        ppo_steps1 = int(500000)
        ppo_steps2 = int(2000000)
        return TrainingPipeline(
            named_losses=dict(
                ppo_loss=PPO(clip_param=0.2, value_loss_coef=0.5, entropy_coef=0.15,),
                ppo_loss2=PPO(clip_param=0.2, value_loss_coef=0.5, entropy_coef=0.001,),
            ),  # type:ignore
            pipeline_stages=[
                PipelineStage(loss_names=["ppo_loss"], max_stage_steps=ppo_steps1),
                PipelineStage(loss_names=["ppo_loss2"], max_stage_steps=ppo_steps2),
            ],
            optimizer_builder=Builder(optim.Adam, dict(lr=1e-3)),
            num_mini_batch=1,
            update_repeats=80,
            max_grad_norm=100,
            num_steps=2000,
            gamma=0.99,
            use_gae=False,
            gae_lambda=0.95,
            advance_scene_rollout_period=None,
            save_interval=200000,
            metric_accumulate_interval=50000,
            lr_scheduler_builder=Builder(
                LambdaLR,
                {
                    "lr_lambda": LinearDecay(steps=ppo_steps1 + ppo_steps2)
                },  # type:ignore
            ),
        )
