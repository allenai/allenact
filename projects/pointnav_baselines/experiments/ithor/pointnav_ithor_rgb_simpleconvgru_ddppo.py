from projects.pointnav_baselines.experiments.ithor.pointnav_ithor_base_ddppo import (
    PointNaviThorBasePPOConfig,
)

from plugins.ithor_plugin.ithor_sensors import RGBSensorThor
from plugins.robothor_plugin.robothor_sensors import GPSCompassSensorRoboThor
from projects.pointnav_baselines.experiments.ithor.pointnav_ithor_base import (
    PointNaviThorBaseConfig,
)
from projects.pointnav_baselines.experiments.pointnav_mixin_simpleconvgru import (
    PointNavMixInSimpleConvGRUConfig,
)
from projects.pointnav_baselines.experiments.pointnav_thor_mixin_ddppo import (
    PointNavThorMixInPPOConfig,
)


class PointNaviThorRGBPPOExperimentConfig(
    PointNaviThorBaseConfig,
    PointNavThorMixInPPOConfig,
    PointNavMixInSimpleConvGRUConfig,
):
    """An Point Navigation experiment configuration in iThor with RGB input."""

    SENSORS = [
        RGBSensorThor(
            height=PointNaviThorBasePPOConfig.SCREEN_SIZE,
            width=PointNaviThorBasePPOConfig.SCREEN_SIZE,
            use_resnet_normalization=True,
            uuid="rgb_lowres",
        ),
        GPSCompassSensorRoboThor(),
    ]

    @classmethod
    def tag(cls):
        return "Pointnav-iTHOR-RGB-SimpleConv-DDPPO"
