from abc import ABC

import gym
import torch.nn as nn

from core.base_abstractions.sensor import DepthSensor, RGBSensor
from plugins.habitat_plugin.habitat_sensors import TargetCoordinatesSensorHabitat
from plugins.robothor_plugin.robothor_sensors import GPSCompassSensorRoboThor
from plugins.robothor_plugin.robothor_tasks import PointNavTask
from projects.pointnav_baselines.experiments.pointnav_base import PointNavBaseConfig
from projects.pointnav_baselines.models.point_nav_models import (
    PointNavActorCriticSimpleConvRNN,
)


class PointNavMixInSimpleConvGRUConfig(PointNavBaseConfig, ABC):
    """The base config for all iTHOR PPO PointNav experiments."""

    @classmethod
    def create_model(cls, **kwargs) -> nn.Module:
        rgb_uuid = next((s.uuid for s in cls.SENSORS if isinstance(s, RGBSensor)), None)
        depth_uuid = next(
            (s.uuid for s in cls.SENSORS if isinstance(s, DepthSensor)), None
        )
        goal_sensor_uuid = next(
            (
                s.uuid
                for s in cls.SENSORS
                if isinstance(
                    s, (GPSCompassSensorRoboThor, TargetCoordinatesSensorHabitat)
                )
            ),
            None,
        )

        return PointNavActorCriticSimpleConvRNN(
            action_space=gym.spaces.Discrete(len(PointNavTask.class_action_names())),
            observation_space=kwargs["sensor_preprocessor_graph"].observation_spaces,
            rgb_uuid=rgb_uuid,
            depth_uuid=depth_uuid,
            goal_sensor_uuid=goal_sensor_uuid,
            hidden_size=512,
            embed_coordinates=False,
            coordinate_dims=2,
            num_rnn_layers=1,
            rnn_type="GRU",
        )
