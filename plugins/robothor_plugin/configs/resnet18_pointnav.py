import gym
import torch.nn as nn

from .resnet18_nav_base import Resnet18NavBaseConfig
from .pointnav_base import PointNavBaseConfig, PointNavTask
from projects.pointnav_baselines.models.point_nav_models import (
    ResnetTensorPointNavActorCritic,
)


class Resnet18PointNavExperimentConfig(PointNavBaseConfig, Resnet18NavBaseConfig):
    """A Point Navigation experiment configuration."""

    OBSERVATIONS = [
        Resnet18NavBaseConfig.RESNET_OUTPUT_UUID,
        PointNavBaseConfig.TARGET_UUID,
    ]

    @classmethod
    def tag(cls):
        return "Resnet18PointNav"

    @classmethod
    def create_model(cls, **kwargs) -> nn.Module:
        return ResnetTensorPointNavActorCritic(
            action_space=gym.spaces.Discrete(len(PointNavTask.class_action_names())),
            observation_space=kwargs["observation_set"].observation_spaces,
            goal_sensor_uuid=PointNavBaseConfig.TARGET_UUID,
            depth_resnet_preprocessor_uuid=Resnet18NavBaseConfig.RESNET_OUTPUT_UUID,
        )
