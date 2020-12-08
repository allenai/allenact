from plugins.ithor_plugin.ithor_sensors import RGBSensorThor
from plugins.robothor_plugin.robothor_sensors import GPSCompassSensorRoboThor
from projects.pointnav_baselines.experiments.ithor.pointnav_ithor_ddppo_base import (
    PointNaviThorPPOBaseConfig,
)
from projects.pointnav_baselines.experiments.ithor.pointnav_ithor_simpleconvgru_base import (
    PointNaviThorSimpleConvGRUBaseConfig,
)


class PointNaviThorRGBPPOExperimentConfig(
    PointNaviThorPPOBaseConfig, PointNaviThorSimpleConvGRUBaseConfig
):
    """An Point Navigation experiment configuration in iThor with RGB input."""

    SENSORS = [
        RGBSensorThor(
            height=PointNaviThorPPOBaseConfig.SCREEN_SIZE,
            width=PointNaviThorPPOBaseConfig.SCREEN_SIZE,
            use_resnet_normalization=True,
            uuid="rgb_lowres",
        ),
        GPSCompassSensorRoboThor(),
    ]

    @classmethod
    def tag(cls):
        return "Pointnav-iTHOR-RGB-SimpleConv-DDPPO"
