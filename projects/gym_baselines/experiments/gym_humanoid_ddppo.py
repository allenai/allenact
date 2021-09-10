from abc import ABC
from typing import cast

import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

from allenact.algorithms.onpolicy_sync.losses.ppo import PPO

from allenact.utils.experiment_utils import (
    TrainingPipeline,
    Builder,
    PipelineStage,
    LinearDecay,
)

from projects.gym_baselines.experiments.gym_humanoid_base import GymHumanoidBaseConfig


class GymHumanoidPPOConfig(GymHumanoidBaseConfig, ABC):
    @classmethod
    def training_pipeline(cls, **kwargs) -> TrainingPipeline:
        lr = 1e-4
        ppo_steps = int(8e7)  # convergence may be after 1e8
        clip_param = 0.1
        value_loss_coef = 0.5
        entropy_coef = 0.0
        num_mini_batch = 4  # optimal 64
        update_repeats = 10
        max_grad_norm = 0.5
        num_steps = 2048
        gamma = 0.99
        use_gae = True
        gae_lambda = 0.95
        advance_scene_rollout_period = None
        save_interval = 200000
        metric_accumulate_interval = 50000
        return TrainingPipeline(
            named_losses=dict(
                ppo_loss=PPO(
                    clip_param=clip_param,
                    value_loss_coef=value_loss_coef,
                    entropy_coef=entropy_coef,
                ),
            ),  # type:ignore
            pipeline_stages=[
                PipelineStage(loss_names=["ppo_loss"], max_stage_steps=ppo_steps),
            ],
            optimizer_builder=Builder(cast(optim.Optimizer, optim.Adam), dict(lr=lr)),
            num_mini_batch=num_mini_batch,
            update_repeats=update_repeats,
            max_grad_norm=max_grad_norm,
            num_steps=num_steps,
            gamma=gamma,
            use_gae=use_gae,
            gae_lambda=gae_lambda,
            advance_scene_rollout_period=advance_scene_rollout_period,
            save_interval=save_interval,
            metric_accumulate_interval=metric_accumulate_interval,
            lr_scheduler_builder=Builder(
                LambdaLR,
                {
                    "lr_lambda": LinearDecay(steps=ppo_steps, startp=1, endp=1)
                },  # constant learning rate
            ),
        )
