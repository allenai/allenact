import copy

import gym
import torch.nn as nn
from torchvision import models

from models.point_nav_models import PointNavActorCriticResNet50GRU
from rl_base.sensor import SensorSuite
from rl_robothor.robothor_tasks import PointNavTask
from rl_robothor.robothor_sensors import GPSCompassSensorRoboThor, DepthSensorRoboThor
# from rl_ai2thor.ai2thor_sensors import RGBSensorThor
from rl_habitat.habitat_preprocessors import ResnetPreProcessorHabitat
from .pointnav_robothor_base import PointNavRoboThorBaseExperimentConfig


class PointNavRoboThorDepthDeterministicResNet50GRUPPOExperimentConfig(PointNavRoboThorBaseExperimentConfig):
    """A Point Navigation experiment configuraqtion in RoboThor"""

    SENSORS = [
        # RGBSensorThor(
        #     {
        #         "height": PointNavRoboThorBaseExperimentConfig.SCREEN_SIZE,
        #         "width": PointNavRoboThorBaseExperimentConfig.SCREEN_SIZE,
        #         "use_resnet_normalization": True,
        #     }
        # ),
        DepthSensorRoboThor(
            {
                "height": PointNavRoboThorBaseExperimentConfig.SCREEN_SIZE,
                "width": PointNavRoboThorBaseExperimentConfig.SCREEN_SIZE,
                "use_resnet_normalization": True,
            }
        ),
        GPSCompassSensorRoboThor({}),
    ]

    PREPROCESSORS = [
        # ResnetPreProcessorHabitat(
        #     config={
        #         "input_height": PointNavRoboThorBaseExperimentConfig.SCREEN_SIZE,
        #         "input_width": PointNavRoboThorBaseExperimentConfig.SCREEN_SIZE,
        #         "output_width": 1,
        #         "output_height": 1,
        #         "output_dims": 2048,
        #         "pool": True,
        #         "torchvision_resnet_model": models.resnet50,
        #         "input_uuids": ["rgb"],
        #         "output_uuid": "rgb_resnet",
        #     }
        # ),
        ResnetPreProcessorHabitat(
            config={
                "input_height": PointNavRoboThorBaseExperimentConfig.SCREEN_SIZE,
                "input_width": PointNavRoboThorBaseExperimentConfig.SCREEN_SIZE,
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
        "target_coordinates_ind",
    ]

    ENV_ARGS = copy.deepcopy(PointNavRoboThorBaseExperimentConfig.ENV_ARGS)
    ENV_ARGS['renderDepthImage'] = True

    @classmethod
    def create_model(cls, **kwargs) -> nn.Module:
        return PointNavActorCriticResNet50GRU(
            action_space=gym.spaces.Discrete(len(PointNavTask.action_names())),
            observation_space=SensorSuite(cls.SENSORS).observation_spaces,
            goal_sensor_uuid="target_coordinates_ind",
            hidden_size=512,
            embed_coordinates=False,
            coordinate_dims=2,
            num_rnn_layers=1,
            rnn_type='GRU'
        )
