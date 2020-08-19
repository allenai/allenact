import gym
import torch.nn as nn

from .nav_base import NavBaseConfig
from .pointnav_base import PointNavBaseConfig, PointNavTask
from projects.pointnav_baselines.models.point_nav_models import (
    PointNavActorCriticSimpleConvRNN,
)
from core.base_abstractions.preprocessor import SensorSuite


class SimplePointNavExperimentConfig(PointNavBaseConfig, NavBaseConfig):
    """A Point Navigation experiment configuration."""

    @classmethod
    def tag(cls):
        return "SimplePointNav"

    @classmethod
    def create_model(cls, **kwargs) -> nn.Module:
        return PointNavActorCriticSimpleConvRNN(
            action_space=gym.spaces.Discrete(len(PointNavTask.class_action_names())),
            observation_space=SensorSuite(cls.SENSORS).observation_spaces,
            goal_sensor_uuid=PointNavBaseConfig.TARGET_UUID,
        )
