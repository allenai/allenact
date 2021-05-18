from allenact_plugins.ithor_plugin.ithor_sensors import GoalObjectTypeThorSensor
from allenact_plugins.robothor_plugin.robothor_sensors import DepthSensorThor
from projects.objectnav_baselines.experiments.ithor.objectnav_ithor_base import (
    ObjectNaviThorBaseConfig,
)
from projects.objectnav_baselines.experiments.objectnav_mixin_ddppo import (
    ObjectNavMixInPPOConfig,
)
from projects.objectnav_baselines.experiments.objectnav_mixin_resnetgru import (
    ObjectNavMixInResNetGRUConfig,
)


class ObjectNaviThorRGBPPOExperimentConfig(
    ObjectNaviThorBaseConfig, ObjectNavMixInPPOConfig, ObjectNavMixInResNetGRUConfig,
):
    """An Object Navigation experiment configuration in iThor with Depth
    input."""

    SENSORS = (
        DepthSensorThor(
            height=ObjectNaviThorBaseConfig.SCREEN_SIZE,
            width=ObjectNaviThorBaseConfig.SCREEN_SIZE,
            use_normalization=True,
            uuid="depth_lowres",
        ),
        GoalObjectTypeThorSensor(object_types=ObjectNaviThorBaseConfig.TARGET_TYPES,),
    )

    @classmethod
    def tag(cls):
        return "Objectnav-iTHOR-Depth-ResNetGRU-DDPPO"
