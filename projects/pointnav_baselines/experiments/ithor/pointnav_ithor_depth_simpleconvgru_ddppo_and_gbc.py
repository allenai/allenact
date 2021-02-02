from allenact_plugins.robothor_plugin.robothor_sensors import (
    DepthSensorThor,
    GPSCompassSensorRoboThor,
)
from projects.pointnav_baselines.experiments.ithor.pointnav_ithor_base import (
    PointNaviThorBaseConfig,
)
from projects.pointnav_baselines.experiments.pointnav_mixin_simpleconvgru import (
    PointNavMixInSimpleConvGRUConfig,
)
from projects.pointnav_baselines.experiments.pointnav_thor_mixin_ddppo_and_gbc import (
    PointNavThorMixInPPOAndGBCConfig,
)


class PointNaviThorDepthPPOExperimentConfig(
    PointNaviThorBaseConfig,
    PointNavThorMixInPPOAndGBCConfig,
    PointNavMixInSimpleConvGRUConfig,
):
    """An Point Navigation experiment configuration in iThor with Depth
    input."""

    SENSORS = PointNavThorMixInPPOAndGBCConfig.SENSORS + (  # type:ignore
        DepthSensorThor(
            height=PointNaviThorBaseConfig.SCREEN_SIZE,
            width=PointNaviThorBaseConfig.SCREEN_SIZE,
            use_normalization=True,
            uuid="depth_lowres",
        ),
        GPSCompassSensorRoboThor(),
    )

    @classmethod
    def tag(cls):
        return "Pointnav-iTHOR-Depth-SimpleConv-DDPPOAndGBC"
