import torch
import torch.optim as optim

from extensions.rl_lighthouse.lighthouse_experiments.base import (
    BaseLightHouseExperimentConfig,
)
from onpolicy_sync.losses import PPO
from onpolicy_sync.losses.imitation import Imitation
from onpolicy_sync.losses.ppo import PPOConfig
from utils.experiment_utils import Builder, PipelineStage, TrainingPipeline, LinearDecay


class LightHouseImitationPPOExperimentConfig(BaseLightHouseExperimentConfig):
    """PPO and Imitation jointly."""

    @classmethod
    def tag(cls):
        return "LightHouseImitationAndPPO"

    @classmethod
    def training_pipeline(cls, **kwargs):
        training_steps = int(3e5)
        lr = 1e-2
        num_mini_batch = 2
        update_repeats = 4
        num_steps = 100
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
                "imitation_loss": Builder(Imitation,),
                "ppo_loss": Builder(
                    PPO,
                    kwargs={"clip_decay": LinearDecay(training_steps)},
                    default=PPOConfig,
                ),
            },
            gamma=gamma,
            use_gae=use_gae,
            gae_lambda=gae_lambda,
            advance_scene_rollout_period=None,
            should_log=False if torch.cuda.is_available() else True,
            pipeline_stages=[
                PipelineStage(
                    loss_names=["imitation_loss", "ppo_loss"],
                    early_stopping_criterion=cls.get_early_stopping_criterion(),
                    max_stage_steps=training_steps,
                ),
            ],
        )
