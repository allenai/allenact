from typing import Dict, Any, List, Optional

import gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

from models.basic_models import LinearActorCritic
from onpolicy_sync.losses import PPO
from onpolicy_sync.losses.ppo import PPOConfig
from rl_base.experiment_config import ExperimentConfig
from rl_base.sensor import SensorSuite
from rl_base.task import TaskSampler
from rl_lighthouse.lighthouse_sensors import FactorialDesignCornerSensor
from rl_lighthouse.lighthouse_tasks import FindGoalLightHouseTaskSampler
from utils.experiment_utils import Builder, PipelineStage, TrainingPipeline, LinearDecay


class LightHouseOneDimPPOExperimentConfig(ExperimentConfig):
    """Find goal in 1-dim lighthouse env.

    Training with PPO.
    """

    WORLD_DIM = 1
    VIEW_RADIUS = 1
    WORLD_RADIUS = 50
    DEGREE = 2
    MAX_STEPS = 1000

    SENSORS = [
        FactorialDesignCornerSensor(
            {"view_radius": VIEW_RADIUS, "world_dim": WORLD_DIM, "degree": DEGREE}
        ),
    ]

    @classmethod
    def tag(cls):
        return "LightHouseOneDimPPO"

    @classmethod
    def training_pipeline(cls, **kwargs):
        ppo_steps = int(1e6)
        lr = 2.5e-4
        num_mini_batch = 6
        update_repeats = 4
        num_steps = 128
        log_interval = cls.MAX_STEPS * 10  # Log every 10 max length tasks
        save_interval = 500000
        gamma = 0.99
        use_gae = True
        gae_lambda = 1.0
        max_grad_norm = 0.5

        return TrainingPipeline(
            save_interval=save_interval,
            log_interval=log_interval,
            optimizer_builder=Builder(optim.Adam, dict(lr=lr)),
            num_mini_batch=num_mini_batch,
            update_repeats=update_repeats,
            max_grad_norm=max_grad_norm,
            num_steps=num_steps,
            named_losses={
                "ppo_loss": Builder(
                    PPO,
                    kwargs={"clip_decay": LinearDecay(ppo_steps)},
                    default=PPOConfig,
                ),
            },
            gamma=gamma,
            use_gae=use_gae,
            gae_lambda=gae_lambda,
            advance_scene_rollout_period=None,
            pipeline_stages=[
                PipelineStage(loss_names=["ppo_loss"], end_criterion=ppo_steps,),
            ],
            lr_scheduler_builder=Builder(
                LambdaLR, {"lr_lambda": LinearDecay(steps=ppo_steps)}
            ),
        )

    @classmethod
    def machine_params(cls, mode="train", **kwargs):
        if mode == "train":
            nprocesses = 6 if not torch.cuda.is_available() else 20
            gpu_ids = [] if not torch.cuda.is_available() else [0]
        elif mode == "valid":
            nprocesses = 1
            gpu_ids = [] if not torch.cuda.is_available() else [1]
        elif mode == "test":
            nprocesses = 1
            gpu_ids = [] if not torch.cuda.is_available() else [0]
        else:
            raise NotImplementedError("mode must be 'train', 'valid', or 'test'.")

        return {"nprocesses": nprocesses, "gpu_ids": gpu_ids}

    @classmethod
    def create_model(cls, **kwargs) -> nn.Module:
        return LinearActorCritic(
            input_key=cls.SENSORS[0]._get_uuid(),
            action_space=gym.spaces.Discrete(2 * cls.WORLD_DIM),
            observation_space=SensorSuite(cls.SENSORS).observation_spaces,
        )

    @staticmethod
    def make_sampler_fn(**kwargs) -> TaskSampler:
        return FindGoalLightHouseTaskSampler(**kwargs)

    def train_task_sampler_args(
        self,
        process_ind: int,
        total_processes: int,
        devices: Optional[List[int]] = None,
        seeds: Optional[List[int]] = None,
        deterministic_cudnn: bool = False,
    ) -> Dict[str, Any]:
        return {
            "world_dim": self.WORLD_DIM,
            "world_radius": self.WORLD_RADIUS,
            "max_steps": self.MAX_STEPS,
            "sensors": self.SENSORS,
            "action_space": gym.spaces.Discrete(2 * self.WORLD_DIM),
            "seed": seeds[process_ind] if seeds is not None else None,
        }

    def valid_task_sampler_args(
        self,
        process_ind: int,
        total_processes: int,
        devices: Optional[List[int]],
        seeds: Optional[List[int]] = None,
        deterministic_cudnn: bool = False,
    ) -> Dict[str, Any]:
        return self.train_task_sampler_args(
            process_ind=process_ind,
            total_processes=total_processes,
            devices=devices,
            seeds=seeds,
            deterministic_cudnn=deterministic_cudnn,
        )

    def test_task_sampler_args(
            self,
            process_ind: int,
            total_processes: int,
            devices: Optional[List[int]],
            seeds: Optional[List[int]] = None,
            deterministic_cudnn: bool = False,
    ) -> Dict[str, Any]:
        return self.train_task_sampler_args(
            process_ind=process_ind,
            total_processes=total_processes,
            devices=devices,
            seeds=seeds,
            deterministic_cudnn=deterministic_cudnn,
        )