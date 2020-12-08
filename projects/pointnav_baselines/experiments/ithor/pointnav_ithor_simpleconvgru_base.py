from abc import ABC

import gym
import torch.nn as nn

from plugins.ithor_plugin.ithor_sensors import RGBSensorThor
from plugins.robothor_plugin.robothor_sensors import (
    DepthSensorThor,
    GPSCompassSensorRoboThor,
)
from plugins.robothor_plugin.robothor_tasks import PointNavTask
from projects.pointnav_baselines.experiments.ithor.pointnav_ithor_base import (
    PointNaviThorBaseConfig,
)
from projects.pointnav_baselines.models.point_nav_models import (
    PointNavActorCriticSimpleConvRNN,
)


class PointNaviThorSimpleConvGRUBaseConfig(PointNaviThorBaseConfig, ABC):
    """The base config for all iTHOR PPO PointNav experiments."""

    @classmethod
    def create_model(cls, **kwargs) -> nn.Module:
        rgb_uuid = next(
            (s.uuid for s in cls.SENSORS if isinstance(s, RGBSensorThor)), None
        )
        depth_uuid = next(
            (s.uuid for s in cls.SENSORS if isinstance(s, DepthSensorThor)), None
        )
        goal_sensor_uuid = next(
            (s.uuid for s in cls.SENSORS if isinstance(s, GPSCompassSensorRoboThor)),
            None,
        )

        return PointNavActorCriticSimpleConvRNN(
            action_space=gym.spaces.Discrete(len(PointNavTask.class_action_names())),
            observation_space=kwargs["observation_set"].observation_spaces,
            rgb_uuid=rgb_uuid,
            depth_uuid=depth_uuid,
            goal_sensor_uuid=goal_sensor_uuid,
            hidden_size=512,
            embed_coordinates=False,
            coordinate_dims=2,
            num_rnn_layers=1,
            rnn_type="GRU",
        )
