import gym
import torch.nn as nn

from .nav_base import NavBaseConfig
from .objectnav_base import ObjectNavBaseConfig, ObjectNavTask
from projects.objectnav_baselines.models.object_nav_models import (
    ObjectNavBaselineActorCritic,
)
from core.base_abstractions.preprocessor import SensorSuite


class SimpleObjectNavExperimentConfig(ObjectNavBaseConfig, NavBaseConfig):
    """A Point Navigation experiment configuration."""

    @classmethod
    def tag(cls):
        return "SimpleObjectNav"

    @classmethod
    def create_model(cls, **kwargs) -> nn.Module:
        return ObjectNavBaselineActorCritic(
            action_space=gym.spaces.Discrete(len(ObjectNavTask.class_action_names())),
            observation_space=SensorSuite(cls.SENSORS).observation_spaces,
            goal_sensor_uuid=ObjectNavBaseConfig.TARGET_UUID,
        )
