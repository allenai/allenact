from allenact_plugins.ithor_plugin.ithor_sensors import GoalObjectTypeThorSensor
from allenact_plugins.robothor_plugin.robothor_sensors import DepthSensorThor
from projects.objectnav_baselines.experiments.objectnav_mixin_ddppo import (
    ObjectNavMixInPPOConfig,
)
from projects.objectnav_baselines.experiments.objectnav_mixin_resnetgru import (
    ObjectNavMixInResNetGRUConfig,
)
from projects.objectnav_baselines.experiments.robothor.objectnav_robothor_base import (
    ObjectNavRoboThorBaseConfig,
)


class ObjectNavRoboThorRGBPPOExperimentConfig(
    ObjectNavRoboThorBaseConfig, ObjectNavMixInPPOConfig, ObjectNavMixInResNetGRUConfig,
):
    """An Object Navigation experiment configuration in RoboThor with Depth
    input."""

    SENSORS = (
        DepthSensorThor(
            height=ObjectNavRoboThorBaseConfig.SCREEN_SIZE,
            width=ObjectNavRoboThorBaseConfig.SCREEN_SIZE,
            use_normalization=True,
            uuid="depth_lowres",
        ),
        GoalObjectTypeThorSensor(
            object_types=ObjectNavRoboThorBaseConfig.TARGET_TYPES,
        ),
    )

    @classmethod
    def tag(cls):
        return "Objectnav-RoboTHOR-Depth-ResNetGRU-DDPPO"
