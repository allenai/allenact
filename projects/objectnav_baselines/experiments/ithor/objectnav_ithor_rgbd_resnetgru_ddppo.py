from allenact_plugins.ithor_plugin.ithor_sensors import (
    RGBSensorThor,
    GoalObjectTypeThorSensor,
)
from allenact_plugins.robothor_plugin.robothor_sensors import DepthSensorThor
from projects.objectnav_baselines.experiments.ithor.objectnav_ithor_base import (
    ObjectNaviThorBaseConfig,
)
from projects.objectnav_baselines.experiments.objectnav_mixin_ddppo import (
    ObjectNavMixInPPOConfig,
)
from projects.objectnav_baselines.experiments.objectnav_mixin_resnet18gru import (
    ObjectNavMixInResNet18GRUConfig,
)


class ObjectNaviThorRGBDPPOExperimentConfig(
    ObjectNaviThorBaseConfig, ObjectNavMixInPPOConfig, ObjectNavMixInResNet18GRUConfig
):
    """An Object Navigation experiment configuration in iTHOR with RGBD
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
