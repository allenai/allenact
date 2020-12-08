from plugins.robothor_plugin.robothor_sensors import (
    DepthSensorThor,
    GPSCompassSensorRoboThor,
)
from projects.pointnav_baselines.experiments.ithor.pointnav_ithor_base import (
    PointNaviThorBaseConfig,
)
from projects.pointnav_baselines.experiments.ithor.pointnav_ithor_ddppo_and_gbc_base import (
    PointNaviThorPPOAndGBCBaseConfig,
)
from projects.pointnav_baselines.experiments.ithor.pointnav_ithor_simpleconvgru_base import (
    PointNaviThorSimpleConvGRUBaseConfig,
)


class PointNaviThorDepthPPOExperimentConfig(
    PointNaviThorPPOAndGBCBaseConfig, PointNaviThorSimpleConvGRUBaseConfig
):
    """An Point Navigation experiment configuration in iThor with Depth
    input."""

    SENSORS = PointNaviThorPPOAndGBCBaseConfig.SENSORS + (
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
