from typing import Optional

import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

from extensions.rl_lighthouse.lighthouse_experiments.base import (
    BaseLightHouseExperimentConfig,
)
from onpolicy_sync.losses import PPO
from onpolicy_sync.losses.ppo import PPOConfig
from utils.experiment_utils import Builder, PipelineStage, TrainingPipeline, LinearDecay


class LightHouseOneDimPPOExperimentConfig(BaseLightHouseExperimentConfig):
    """Find goal in 1-dim lighthouse env.

    Training with PPO.
    """

    WORLD_DIM = 1
    VIEW_RADIUS = 1
    WORLD_RADIUS = 10
    DEGREE = -1
    MAX_STEPS = 1000
    GPU_ID: Optional[int] = None

    @classmethod
    def tag(cls):
        return "LightHouseOneDimPPO"

    @classmethod
    def training_pipeline(cls, **kwargs):
        ppo_steps = int(1e6)
        lr = 2.5e-4
        num_mini_batch = 2
        update_repeats = 4
        num_steps = 128
        metric_accumulate_interval = cls.MAX_STEPS * 10  # Log every 10 max length tasks
        save_interval = 500000
        gamma = 0.99
        use_gae = True
        gae_lambda = 1.0
        max_grad_norm = 0.5

        return TrainingPipeline(
            save_interval=save_interval,
            metric_accumulate_interval=metric_accumulate_interval,
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
                PipelineStage(loss_names=["ppo_loss"], max_stage_steps=ppo_steps,),
            ],
            lr_scheduler_builder=Builder(
                LambdaLR, {"lr_lambda": LinearDecay(steps=ppo_steps)}
            ),
        )
