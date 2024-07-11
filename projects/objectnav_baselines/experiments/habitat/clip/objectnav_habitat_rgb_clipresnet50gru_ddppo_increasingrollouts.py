import torch
import torch.optim as optim

from allenact.algorithms.onpolicy_sync.losses import PPO
from allenact.algorithms.onpolicy_sync.losses.ppo import PPOConfig
from allenact.utils.experiment_utils import (
    Builder,
    TrainingPipeline,
    PipelineStage,
    TrainingSettings,
)
from projects.objectnav_baselines.experiments.habitat.clip.objectnav_habitat_rgb_clipresnet50gru_ddppo import (
    ObjectNavHabitatRGBClipResNet50GRUDDPPOExperimentConfig,
)
from projects.objectnav_baselines.mixins import update_with_auxiliary_losses


class ObjectNavHabitatRGBClipResNet50GRUDDPPOIncreasingLengthExpConfig(
    ObjectNavHabitatRGBClipResNet50GRUDDPPOExperimentConfig
):
    def __init__(self, lr=1e-4, **kwargs):
        super().__init__(lr, **kwargs)
        self.lr = lr

    def training_pipeline(self, **kwargs) -> TrainingPipeline:
        auxiliary_uuids = self.auxiliary_uuids
        multiple_beliefs = False
        normalize_advantage = False
        advance_scene_rollout_period = self.ADVANCE_SCENE_ROLLOUT_PERIOD
        log_interval_small = (
            self.num_train_processes * 32 * 10 if torch.cuda.is_available() else 1
        )
        log_interval_med = (
            self.num_train_processes * 64 * 5 if torch.cuda.is_available() else 1
        )
        log_interval_large = (
            self.num_train_processes * 128 * 5 if torch.cuda.is_available() else 1
        )

        batch_steps_0 = int(10e6)
        batch_steps_1 = int(10e6)
        batch_steps_2 = int(1e9) - batch_steps_0 - batch_steps_1

        lr = self.lr
        num_mini_batch = 1
        update_repeats = 4
        save_interval = 5000000
        gamma = 0.99
        use_gae = True
        gae_lambda = 0.95
        max_grad_norm = 0.5

        named_losses = {
            "ppo_loss": (PPO(**PPOConfig, normalize_advantage=normalize_advantage), 1.0)
        }
        named_losses = update_with_auxiliary_losses(
            named_losses=named_losses,
            auxiliary_uuids=auxiliary_uuids,
            multiple_beliefs=multiple_beliefs,
        )

        return TrainingPipeline(
            save_interval=save_interval,
            optimizer_builder=Builder(optim.Adam, dict(lr=lr)),
            num_mini_batch=num_mini_batch,
            update_repeats=update_repeats,
            max_grad_norm=max_grad_norm,
            named_losses={key: val[0] for key, val in named_losses.items()},
            gamma=gamma,
            use_gae=use_gae,
            gae_lambda=gae_lambda,
            advance_scene_rollout_period=advance_scene_rollout_period,
            pipeline_stages=[
                PipelineStage(
                    loss_names=list(named_losses.keys()),
                    max_stage_steps=batch_steps_0,
                    training_settings=TrainingSettings(
                        num_steps=32, metric_accumulate_interval=log_interval_small
                    ),
                ),
                PipelineStage(
                    loss_names=list(named_losses.keys()),
                    max_stage_steps=batch_steps_1,
                    training_settings=TrainingSettings(
                        num_steps=64,
                        metric_accumulate_interval=log_interval_med,
                    ),
                ),
                PipelineStage(
                    loss_names=list(named_losses.keys()),
                    max_stage_steps=batch_steps_2,
                    training_settings=TrainingSettings(
                        num_steps=128,
                        metric_accumulate_interval=log_interval_large,
                    ),
                ),
            ],
            lr_scheduler_builder=None,
        )

    def tag(self):
        return (
            super(
                ObjectNavHabitatRGBClipResNet50GRUDDPPOIncreasingLengthExpConfig, self
            )
            .tag()
            .replace("-DDPPO-lr", "-DDPPO-IncRollouts-lr")
        )
