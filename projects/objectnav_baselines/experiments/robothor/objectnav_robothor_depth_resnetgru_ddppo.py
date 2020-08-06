import gym
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torchvision import models

from common.algorithms.onpolicy_sync.losses import PPO
from common.algorithms.onpolicy_sync.losses.ppo import PPOConfig
from projects.objectnav_baselines.experiments.robothor.objectnav_robothor_base import (
    ObjectNavRoboThorBaseConfig,
)
from projects.objectnav_baselines.models.object_nav_models import (
    ResnetTensorObjectNavActorCritic,
)
from plugins.ithor_plugin.ithor_sensors import GoalObjectTypeThorSensor
from plugins.habitat_plugin.habitat_preprocessors import ResnetPreProcessorHabitat
from plugins.robothor_plugin.robothor_sensors import DepthSensorRoboThor
from plugins.robothor_plugin.robothor_tasks import ObjectNavTask
from utils.experiment_utils import Builder, PipelineStage, TrainingPipeline, LinearDecay


class ObjectNavRoboThorRGBPPOExperimentConfig(ObjectNavRoboThorBaseConfig):
    """An Object Navigation experiment configuration in RoboThor with Depth
    input."""

    def __init__(self):
        super().__init__()

        self.ENV_ARGS["renderDepthImage"] = True

        DepthSensorRoboThor(
            height=self.SCREEN_SIZE,
            width=self.SCREEN_SIZE,
            use_resnet_normalization=True,
            uuid="depth_lowres",
        ),
        GoalObjectTypeThorSensor(object_types=self.TARGET_TYPES,),

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
                    "input_uuids": ["depth_lowres"],
                    "output_uuid": "depth_resnet",
                    "parallel": False,  # TODO False for debugging
                },
            ),
        ]

        self.OBSERVATIONS = [
            "depth_resnet",
            "goal_object_type_ind",
        ]

    @classmethod
    def tag(cls):
        return "Objectnav-RoboTHOR-Depth-ResNetGRU-DDPPO"

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
            named_losses={"ppo_loss": Builder(PPO, kwargs={}, default=PPOConfig,)},
            gamma=gamma,
            use_gae=use_gae,
            gae_lambda=gae_lambda,
            advance_scene_rollout_period=self.ADVANCE_SCENE_ROLLOUT_PERIOD,
            pipeline_stages=[
                PipelineStage(loss_names=["ppo_loss"], max_stage_steps=ppo_steps)
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
            depth_resnet_preprocessor_uuid="depth_resnet",
            hidden_size=512,
            goal_dims=32,
        )
