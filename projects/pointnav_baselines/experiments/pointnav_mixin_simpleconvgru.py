from abc import ABC

import gym
import torch.nn as nn

from allenact.embodiedai.sensors.vision_sensors import RGBSensor, DepthSensor

# fmt: off
try:
    # Habitat may not be installed, just create a fake class here in that case
    from allenact_plugins.habitat_plugin.habitat_sensors import TargetCoordinatesSensorHabitat
except ImportError:
    class TargetCoordinatesSensorHabitat:  #type:ignore
        pass
# fmt: on

from allenact_plugins.robothor_plugin.robothor_sensors import GPSCompassSensorRoboThor
from allenact_plugins.robothor_plugin.robothor_tasks import PointNavTask
from projects.pointnav_baselines.experiments.pointnav_base import PointNavBaseConfig
from projects.pointnav_baselines.models.point_nav_models import PointNavActorCritic


class PointNavMixInSimpleConvGRUConfig(PointNavBaseConfig, ABC):
    """The base config for all iTHOR PPO PointNav experiments."""

    # TODO only tested in roboTHOR Depth
    BACKBONE = (  # choose one
        "gnresnet18"
        # "simple_cnn"
    )

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

        return PointNavActorCritic(
            # Env and Tak
            action_space=gym.spaces.Discrete(len(PointNavTask.class_action_names())),
            observation_space=kwargs["sensor_preprocessor_graph"].observation_spaces,
            rgb_uuid=rgb_uuid,
            depth_uuid=depth_uuid,
            goal_sensor_uuid=goal_sensor_uuid,
            # RNN
            hidden_size=228
            if cls.MULTIPLE_BELIEFS and len(cls.AUXILIARY_UUIDS) > 1
            else 512,
            num_rnn_layers=1,
            rnn_type="GRU",
            add_prev_actions=cls.ADD_PREV_ACTIONS,
            action_embed_size=4,
            # CNN
            backbone=cls.BACKBONE,
            resnet_baseplanes=32,
            embed_coordinates=False,
            coordinate_dims=2,
            # Aux
            auxiliary_uuids=cls.AUXILIARY_UUIDS,
            multiple_beliefs=cls.MULTIPLE_BELIEFS,
            beliefs_fusion=cls.BELIEF_FUSION,
        )
