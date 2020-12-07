from plugins.ithor_plugin.ithor_sensors import RGBSensorThor
from plugins.robothor_plugin.robothor_sensors import (
    DepthSensorThor,
    GPSCompassSensorRoboThor,
)
from projects.pointnav_baselines.experiments.ithor.pointnav_ithor_base import (
    PointNaviThorBaseConfig,
)
from projects.pointnav_baselines.experiments.ithor.pointnav_ithor_simpleconvgru_ddppo_base import (
    PointNaviThorSimpleConvGRUPPOBaseConfig,
)


class PointNaviThorRGBDPPOExperimentConfig(PointNaviThorSimpleConvGRUPPOBaseConfig):
    """An Point Navigation experiment configuration in iThor with RGBD
    input."""

    SENSORS = [
        RGBSensorThor(
            height=PointNaviThorBaseConfig.SCREEN_SIZE,
            width=PointNaviThorBaseConfig.SCREEN_SIZE,
            use_resnet_normalization=True,
            uuid="rgb_lowres",
        ),
        DepthSensorThor(
            height=PointNaviThorBaseConfig.SCREEN_SIZE,
            width=PointNaviThorBaseConfig.SCREEN_SIZE,
            use_normalization=True,
            uuid="depth_lowres",
        ),
        GPSCompassSensorRoboThor(),
    ]

    @classmethod
    def tag(cls):
        return "Pointnav-iTHOR-RGBD-SimpleConv-DDPPO"
