import typing
from typing import Optional, List, Any, Dict

import gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torchvision import models

from core.algorithms.onpolicy_sync.losses import PPO
from core.algorithms.onpolicy_sync.losses.abstract_loss import AbstractActorCriticLoss
from core.algorithms.onpolicy_sync.losses.ppo import PPOConfig
from core.algorithms.onpolicy_sync.policy import ObservationType
from core.base_abstractions.distributions import CategoricalDistr
from core.base_abstractions.misc import ActorCriticOutput
from core.base_abstractions.sensor import ExpertActionSensor
from plugins.habitat_plugin.habitat_preprocessors import ResnetPreProcessorHabitat
from plugins.ithor_plugin.ithor_sensors import RGBSensorThor, GoalObjectTypeThorSensor
from plugins.robothor_plugin import robothor_constants
from plugins.robothor_plugin.robothor_tasks import ObjectNavTask
from projects.objectnav_baselines.experiments.robothor.objectnav_robothor_base import (
    ObjectNavRoboThorBaseConfig,
)
from projects.objectnav_baselines.models.object_nav_models import (
    ResnetTensorObjectNavActorCritic,
)
from utils.experiment_utils import Builder, PipelineStage, TrainingPipeline, LinearDecay


class YesNoImitation(AbstractActorCriticLoss):
    def __init__(self, yes_action_index: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.yes_action_index = yes_action_index

    def loss(  # type: ignore
        self,
        step_count: int,
        batch: ObservationType,
        actor_critic_output: ActorCriticOutput[CategoricalDistr],
        *args,
        **kwargs
    ):
        observations = typing.cast(Dict[str, torch.Tensor], batch["observations"])

        assert "expert_action" in observations

        expert_actions_and_mask = observations["expert_action"]
        if len(expert_actions_and_mask.shape) == 3:
            # No agent dimension in expert action
            expert_actions_and_mask = expert_actions_and_mask.unsqueeze(-2)

        assert expert_actions_and_mask.shape[-1] == 2
        expert_actions_and_mask_reshaped = expert_actions_and_mask.view(-1, 2)

        expert_actions = expert_actions_and_mask_reshaped[:, 0].view(
            *expert_actions_and_mask.shape[:-1], 1
        )
        expert_actions_masks = (
            expert_actions_and_mask_reshaped[:, 1]
            .float()
            .view(*expert_actions_and_mask.shape[:-1], 1)
        )

        log_probs_yes_action = actor_critic_output.distributions.log_probs(
            typing.cast(
                torch.LongTensor,
                self.yes_action_index + torch.zeros_like(expert_actions),
            )
        )
        log_probs_not_yes_action = torch.log(1.0 - torch.exp(log_probs_yes_action)) # type: ignore
        expert_action_was_yes_action = (
            expert_actions_masks * (expert_actions == self.yes_action_index).float()
        )

        total_loss = -(
            (
                log_probs_yes_action * expert_action_was_yes_action
                + log_probs_not_yes_action * (1.0 - expert_action_was_yes_action) # type: ignore
            )
        ).mean()

        return (total_loss, {"yes_action_cross_entropy": total_loss.item(),})


class ObjectNavThorRGBPPOAndSimpleBCExperimentConfig(ObjectNavRoboThorBaseConfig):
    """An Object Navigation experiment configuration in RoboThor with RGB
    input."""

    def __init__(self):
        super().__init__()

        self.SENSORS = [
            RGBSensorThor(
                height=self.SCREEN_SIZE,
                width=self.SCREEN_SIZE,
                use_resnet_normalization=True,
                uuid="rgb_lowres",
            ),
            GoalObjectTypeThorSensor(object_types=self.TARGET_TYPES,),
            ExpertActionSensor(
                nactions=len(ObjectNavTask.class_action_names()),
                expert_args=dict(end_action_only=True),
            ),
        ]

        self.PREPROCESSORS = [
            Builder(
                ResnetPreProcessorHabitat,
                {
                    "input_height": self.SCREEN_SIZE,
                    "input_width": self.SCREEN_SIZE,
                    "output_width": 7,
                    "output_height": 7,
                    "output_dims": 512,
                    "pool": False,
                    "torchvision_resnet_model": models.resnet18,
                    "input_uuids": ["rgb_lowres"],
                    "output_uuid": "rgb_resnet",
                    "parallel": False,
                },
            ),
        ]

        self.OBSERVATIONS = [
            "rgb_resnet",
            "goal_object_type_ind",
            "expert_action",
        ]

    @classmethod
    def tag(cls):
        return "Objectnav-RoboTHOR-RGB-ResNetGRU-DDPPO-AND-SimpleBC"

    def training_pipeline(self, **kwargs):
        ppo_steps = int(300000000)
        lr = 3e-4
        num_mini_batch = 1
        update_repeats = 3
        num_steps = 30
        save_interval = 5000000
        log_interval = 10000
        gamma = 0.99
        use_gae = True
        gae_lambda = 0.95
        max_grad_norm = 0.5
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
                "yes_no_imitation_loss": YesNoImitation(
                    yes_action_index=ObjectNavTask.class_action_names().index(
                        robothor_constants.END
                    )
                ),
            },
            gamma=gamma,
            use_gae=use_gae,
            gae_lambda=gae_lambda,
            advance_scene_rollout_period=self.ADVANCE_SCENE_ROLLOUT_PERIOD,
            pipeline_stages=[
                PipelineStage(
                    loss_names=["ppo_loss", "yes_no_imitation_loss"],
                    max_stage_steps=ppo_steps,
                ),
            ],
            lr_scheduler_builder=Builder(
                LambdaLR, {"lr_lambda": LinearDecay(steps=ppo_steps)}
            ),
        )

    @classmethod
    def create_model(cls, **kwargs) -> nn.Module:
        return ResnetTensorObjectNavActorCritic(
            action_space=gym.spaces.Discrete(len(ObjectNavTask.class_action_names())),
            observation_space=kwargs["observation_set"].observation_spaces,
            goal_sensor_uuid="goal_object_type_ind",
            rgb_resnet_preprocessor_uuid="rgb_resnet",
            hidden_size=512,
            goal_dims=32,
        )

    def test_task_sampler_args(
        self,
        process_ind: int,
        total_processes: int,
        devices: Optional[List[int]] = None,
        seeds: Optional[List[int]] = None,
        deterministic_cudnn: bool = False,
    ) -> Dict[str, Any]:
        kwargs = super(
            ObjectNavThorRGBPPOAndSimpleBCExperimentConfig, self
        ).test_task_sampler_args(
            process_ind=process_ind,
            total_processes=total_processes,
            devices=devices,
            seeds=seeds,
            deterministic_cudnn=deterministic_cudnn,
        )
        kwargs["rewards_config"]["shaping_weight"] = 0

        return kwargs
