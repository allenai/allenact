from plugins.habitat_plugin.habitat_sensors import (
    RGBSensorHabitat,
    TargetCoordinatesSensorHabitat,
)
from projects.pointnav_baselines.experiments.habitat.pointnav_habitat_base import (
    PointNavHabitatBaseConfig,
)
from projects.pointnav_baselines.experiments.pointnav_habitat_mixin_ddppo import PointNavHabitatMixInPPOConfig
from projects.pointnav_baselines.experiments.pointnav_mixin_simpleconvgru import PointNavMixInSimpleConvGRUConfig


class PointNavHabitatDepthDeterministiSimpleConvGRUDDPPOExperimentConfig(
    PointNavHabitatBaseConfig, PointNavHabitatMixInPPOConfig, PointNavMixInSimpleConvGRUConfig
):
    """An Point Navigation experiment configuration in Habitat with Depth
    input."""

    SENSORS = [
        RGBSensorHabitat(
            height=PointNavHabitatBaseConfig.SCREEN_SIZE,
            width=PointNavHabitatBaseConfig.SCREEN_SIZE,
            use_resnet_normalization=True,
        ),
        TargetCoordinatesSensorHabitat(coordinate_dims=2),
    ]

    @classmethod
    def tag(cls):
        return "Pointnav-Habitat-RGB-SimpleConv-DDPPO"
