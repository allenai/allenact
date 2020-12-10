from plugins.robothor_plugin.robothor_sensors import (
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
    """An Point Navigation experiment configuration in RoboTHOR with Depth
    input."""

    SENSORS = [
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
        return "Pointnav-RoboTHOR-Depth-SimpleConv-DDPPO"
