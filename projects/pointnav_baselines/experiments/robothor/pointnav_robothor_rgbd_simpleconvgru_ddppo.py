from allenact_plugins.ithor_plugin.ithor_sensors import RGBSensorThor
from allenact_plugins.robothor_plugin.robothor_sensors import (
    DepthSensorThor,
    GPSCompassSensorRoboThor,
)
from projects.pointnav_baselines.experiments.pointnav_mixin_simpleconvgru import (
    PointNavMixInSimpleConvGRUConfig,
)
from projects.pointnav_baselines.experiments.pointnav_thor_mixin_ddppo import (
    PointNavThorMixInPPOConfig,
)
from projects.pointnav_baselines.experiments.robothor.pointnav_robothor_base import (
    PointNavRoboThorBaseConfig,
)


class PointNavRoboThorRGBPPOExperimentConfig(
    PointNavRoboThorBaseConfig,
    PointNavThorMixInPPOConfig,
    PointNavMixInSimpleConvGRUConfig,
):
    """An Point Navigation experiment configuration in RoboThor with RGBD
    input."""

    SENSORS = [
        RGBSensorThor(
            height=PointNavRoboThorBaseConfig.SCREEN_SIZE,
            width=PointNavRoboThorBaseConfig.SCREEN_SIZE,
            use_resnet_normalization=True,
            uuid="rgb_lowres",
        ),
        DepthSensorThor(
            height=PointNavRoboThorBaseConfig.SCREEN_SIZE,
            width=PointNavRoboThorBaseConfig.SCREEN_SIZE,
            use_normalization=True,
            uuid="depth_lowres",
        ),
        GPSCompassSensorRoboThor(),
    ]

    @classmethod
    def tag(cls):
        return "Pointnav-RoboTHOR-RGBD-SimpleConv-DDPPO"
