import copy

import gym
import torch.nn as nn
from torchvision import models

from models.object_nav_models import ObjectNavResNetActorCritic
from rl_base.sensor import SensorSuite
from rl_robothor.robothor_tasks import ObjectNavTask
from rl_ai2thor.ai2thor_sensors import RGBSensorThor, GoalObjectTypeThorSensor
from rl_robothor.robothor_sensors import DepthSensorRoboThor
from rl_habitat.habitat_preprocessors import ResnetPreProcessorHabitat
from .objectnav_robothor_base import ObjectNavRoboThorBaseExperimentConfig



class ObjectNavRoboThorDepthDeterministicResnet50PPOExperimentConfig(ObjectNavRoboThorBaseExperimentConfig):
    """An Object Navigation experiment configuration in RoboThor"""

    SENSORS = [
        DepthSensorRoboThor(
            {
                "height": ObjectNavRoboThorBaseExperimentConfig.SCREEN_SIZE,
                "width": ObjectNavRoboThorBaseExperimentConfig.SCREEN_SIZE,
                "use_resnet_normalization": True,
            }
        ),
        GoalObjectTypeThorSensor({"object_types": ObjectNavRoboThorBaseExperimentConfig.OBJECT_TYPES}),
    ]

    PREPROCESSORS = [
        ResnetPreProcessorHabitat(
            config={
                "input_height": ObjectNavRoboThorBaseExperimentConfig.SCREEN_SIZE,
                "input_width": ObjectNavRoboThorBaseExperimentConfig.SCREEN_SIZE,
                "output_width": 1,
                "output_height": 1,
                "output_dims": 2048,
                "pool": True,
                "torchvision_resnet_model": models.resnet50,
                "input_uuids": ["depth"],
                "output_uuid": "depth_resnet",
            }
        ),
    ]

    OBSERVATIONS = [
        "depth_resnet",
        "goal_object_type_ind",
    ]

    ENV_ARGS = copy.deepcopy(ObjectNavRoboThorBaseExperimentConfig.ENV_ARGS)
    ENV_ARGS['renderDepthImage'] = True

    @classmethod
    def create_model(cls, **kwargs) -> nn.Module:
        return ObjectNavResNetActorCritic(
            action_space=gym.spaces.Discrete(len(ObjectNavTask.action_names())),
            observation_space=SensorSuite(cls.SENSORS).observation_spaces,
            goal_sensor_uuid="goal_object_type_ind",
            hidden_size=512,
            object_type_embedding_dim=32,
            num_rnn_layers=1,
            rnn_type='GRU'
        )
