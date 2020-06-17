import copy

import gym
import torch.nn as nn

from models.object_nav_models import ObjectNavBaselineActorCritic
from rl_base.sensor import SensorSuite
from rl_robothor.robothor_tasks import ObjectNavTask
from rl_ai2thor.ai2thor_sensors import RGBSensorThor, GoalObjectTypeThorSensor
from .objectnav_robothor_base import ObjectNavRoboThorBaseExperimentConfig


class ObjectNavRoboThorRGBDeterministicSimpleConvGRUPPOExperimentConfig(ObjectNavRoboThorBaseExperimentConfig):
    """An Object Navigation experiment configuration in RoboThor"""

    SENSORS = [
        RGBSensorThor(
            {
                "height": ObjectNavRoboThorBaseExperimentConfig.SCREEN_SIZE,
                "width": ObjectNavRoboThorBaseExperimentConfig.SCREEN_SIZE,
                "use_resnet_normalization": True,
            }
        ),
        GoalObjectTypeThorSensor({"object_types": ObjectNavRoboThorBaseExperimentConfig.OBJECT_TYPES}),
    ]

    PREPROCESSORS = []

    OBSERVATIONS = [
        "rgb",
        "goal_object_type_ind",
    ]

    ENV_ARGS = copy.deepcopy(ObjectNavRoboThorBaseExperimentConfig.ENV_ARGS)

    @classmethod
    def create_model(cls, **kwargs) -> nn.Module:
        return ObjectNavBaselineActorCritic(
            action_space=gym.spaces.Discrete(len(ObjectNavTask.action_names())),
            observation_space=SensorSuite(cls.SENSORS).observation_spaces,
            goal_sensor_uuid="goal_object_type_ind",
            hidden_size=512,
            object_type_embedding_dim=32,
            num_rnn_layers=1,
            rnn_type='GRU'
        )
