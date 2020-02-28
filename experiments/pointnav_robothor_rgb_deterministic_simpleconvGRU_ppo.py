import copy

import gym
import torch.nn as nn

from models.point_nav_models import PointNavActorCriticSimpleConvLSTM
from rl_base.sensor import SensorSuite
from rl_robothor.robothor_tasks import PointNavTask
from rl_robothor.robothor_sensors import GPSCompassSensorRoboThor
from rl_ai2thor.ai2thor_sensors import RGBSensorThor
from .pointnav_robothor_base import PointNavRoboThorBaseExperimentConfig


class PointNavRoboThorRGBDeterministicSimpleConvGRUPPOExperimentConfig(PointNavRoboThorBaseExperimentConfig):
    """A Point Navigation experiment configuraqtion in RoboThor"""

    SENSORS = [
        RGBSensorThor(
            {
                "height": PointNavRoboThorBaseExperimentConfig.SCREEN_SIZE,
                "width": PointNavRoboThorBaseExperimentConfig.SCREEN_SIZE,
                "use_resnet_normalization": True,
            }
        ),
        GPSCompassSensorRoboThor({}),
    ]

    PREPROCESSORS = []

    OBSERVATIONS = [
        "rgb",
        "target_coordinates_ind",
    ]

    ENV_ARGS = copy.deepcopy(PointNavRoboThorBaseExperimentConfig.ENV_ARGS)

    @classmethod
    def create_model(cls, **kwargs) -> nn.Module:
        return PointNavActorCriticSimpleConvLSTM(
            action_space=gym.spaces.Discrete(len(PointNavTask.action_names())),
            observation_space=SensorSuite(cls.SENSORS).observation_spaces,
            goal_sensor_uuid="target_coordinates_ind",
            hidden_size=512,
            embed_coordinates=False,
            coordinate_dims=2,
            num_rnn_layers=1,
            rnn_type='GRU'
        )
