import gym
import torch.nn as nn

from .resnet18_nav_base import Resnet18NavBaseConfig
from .objectnav_base import ObjectNavBaseConfig, ObjectNavTask
from plugins.robothor_plugin.robothor_models import ResnetTensorObjectNavActorCritic


class Resnet18ObjectNavExperimentConfig(ObjectNavBaseConfig, Resnet18NavBaseConfig):
    """An Object Navigation experiment configuration."""

    OBSERVATIONS = [
        Resnet18NavBaseConfig.RESNET_OUTPUT_UUID,
        ObjectNavBaseConfig.TARGET_UUID,
    ]

    @classmethod
    def tag(cls):
        return "Resnet18ObjectNav"

    @classmethod
    def create_model(cls, **kwargs) -> nn.Module:
        return ResnetTensorObjectNavActorCritic(
            action_space=gym.spaces.Discrete(len(ObjectNavTask.class_action_names())),
            observation_space=kwargs["observation_set"].observation_spaces,
            goal_sensor_uuid=ObjectNavBaseConfig.TARGET_UUID,
            resnet_preprocessor_uuid=Resnet18NavBaseConfig.RESNET_OUTPUT_UUID,
            rnn_hidden_size=512,
            goal_dims=32,
        )
