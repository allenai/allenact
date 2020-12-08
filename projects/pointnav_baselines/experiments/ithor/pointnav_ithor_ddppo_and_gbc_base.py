from abc import ABC

import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

from core.algorithms.onpolicy_sync.losses import PPO
from core.algorithms.onpolicy_sync.losses.grouped_action_imitation import (
    GroupedActionImitation,
)
from core.algorithms.onpolicy_sync.losses.ppo import PPOConfig
from plugins.ithor_plugin.ithor_sensors import TakeEndActionThorNavSensor
from plugins.robothor_plugin import robothor_constants
from plugins.robothor_plugin.robothor_tasks import PointNavTask
from projects.pointnav_baselines.experiments.ithor.pointnav_ithor_base import (
    PointNaviThorBaseConfig,
)
from utils.experiment_utils import Builder, PipelineStage, TrainingPipeline, LinearDecay


class PointNaviThorPPOAndGBCBaseConfig(PointNaviThorBaseConfig, ABC):
    """The base config for all iTHOR PPO PointNav experiments."""

    SENSORS = (
        TakeEndActionThorNavSensor(
            nactions=len(PointNavTask.class_action_names()), uuid="expert_group_action"
        ),
    )

    @classmethod
    def training_pipeline(cls, **kwargs):
        ppo_steps = int(75000000)
        lr = 3e-4
        num_mini_batch = 1
        update_repeats = 4
        num_steps = 128
        save_interval = 5000000
        log_interval = 10000
        gamma = 0.99
        use_gae = True
        gae_lambda = 0.95
        max_grad_norm = 0.5

        action_strs = PointNavTask.class_action_names()
        non_end_action_inds_set = {
            i for i, a in enumerate(action_strs) if a != robothor_constants.END
        }
        end_action_ind_set = {action_strs.index(robothor_constants.END)}

        return TrainingPipeline(
            save_interval=save_interval,
            metric_accumulate_interval=log_interval,
            optimizer_builder=Builder(optim.Adam, dict(lr=lr)),
            num_mini_batch=num_mini_batch,
            update_repeats=update_repeats,
            max_grad_norm=max_grad_norm,
            num_steps=num_steps,
            named_losses={
                "ppo_loss": PPO(**PPOConfig),
                "grouped_action_imitation": GroupedActionImitation(
                    nactions=len(PointNavTask.class_action_names()),
                    action_groups=[non_end_action_inds_set, end_action_ind_set],
                ),
            },
            gamma=gamma,
            use_gae=use_gae,
            gae_lambda=gae_lambda,
            advance_scene_rollout_period=cls.ADVANCE_SCENE_ROLLOUT_PERIOD,
            pipeline_stages=[
                PipelineStage(
                    loss_names=["ppo_loss", "grouped_action_imitation"],
                    max_stage_steps=ppo_steps,
                )
            ],
            lr_scheduler_builder=Builder(
                LambdaLR, {"lr_lambda": LinearDecay(steps=ppo_steps)}
            ),
        )
