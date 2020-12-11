from plugins.ithor_plugin.ithor_sensors import RGBSensorThor, GoalObjectTypeThorSensor
from plugins.robothor_plugin.robothor_sensors import DepthSensorThor
from projects.objectnav_baselines.experiments.ithor.objectnav_ithor_base import (
    ObjectNaviThorBaseConfig,
)
from projects.objectnav_baselines.experiments.objectnav_mixin_ddppo import (
    ObjectNavMixInPPOConfig,
)
from projects.objectnav_baselines.experiments.objectnav_mixin_resnetgru import (
    ObjectNavMixInResNetGRUConfig,
)


class ObjectNavRoboThorRGBPPOExperimentConfig(
    ObjectNaviThorBaseConfig, ObjectNavMixInPPOConfig, ObjectNavMixInResNetGRUConfig
):
    """An Object Navigation experiment configuration in iThor with RGBD
    input."""

    SENSORS = [
        RGBSensorThor(
            height=ObjectNaviThorBaseConfig.SCREEN_SIZE,
            width=ObjectNaviThorBaseConfig.SCREEN_SIZE,
            use_resnet_normalization=True,
            uuid="rgb_lowres",
        ),
        DepthSensorThor(
            height=ObjectNaviThorBaseConfig.SCREEN_SIZE,
            width=ObjectNaviThorBaseConfig.SCREEN_SIZE,
            use_normalization=True,
            uuid="depth_lowres",
        ),
        GoalObjectTypeThorSensor(object_types=ObjectNaviThorBaseConfig.TARGET_TYPES,),
    ]

    @classmethod
    def tag(cls):
        return "Objectnav-iTHOR-RGBD-ResNetGRU-DDPPO"
