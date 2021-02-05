from allenact_plugins.ithor_plugin.ithor_sensors import (
    RGBSensorThor,
    GoalObjectTypeThorSensor,
)
from projects.objectnav_baselines.experiments.objectnav_mixin_ddppo import (
    ObjectNavMixInPPOConfig,
)
from projects.objectnav_baselines.experiments.objectnav_mixin_resnetgru import (
    ObjectNavMixInResNetGRUConfig,
)
from projects.objectnav_baselines.experiments.robothor.objectnav_robothor_base import (
    ObjectNavRoboThorBaseConfig,
)


class ObjectNavThorRGBPPOExperimentConfig(
    ObjectNavRoboThorBaseConfig, ObjectNavMixInPPOConfig, ObjectNavMixInResNetGRUConfig,
):
    """An Object Navigation experiment configuration in RoboThor with RGB
    input."""

    SENSORS = [
        RGBSensorThor(
            height=ObjectNavRoboThorBaseConfig.SCREEN_SIZE,
            width=ObjectNavRoboThorBaseConfig.SCREEN_SIZE,
            use_resnet_normalization=True,
            uuid="rgb_lowres",
        ),
        GoalObjectTypeThorSensor(
            object_types=ObjectNavRoboThorBaseConfig.TARGET_TYPES,
        ),
    ]

    @classmethod
    def tag(cls):
        return "Objectnav-RoboTHOR-RGB-ResNetGRU-DDPPO"
