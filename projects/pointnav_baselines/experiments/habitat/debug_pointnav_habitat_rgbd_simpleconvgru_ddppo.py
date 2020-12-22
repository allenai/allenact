from plugins.habitat_plugin.habitat_sensors import (
    RGBSensorHabitat,
    TargetCoordinatesSensorHabitat,
    DepthSensorHabitat,
)
from projects.pointnav_baselines.experiments.habitat.debug_pointnav_habitat_base import (
    DebugPointNavHabitatBaseConfig,
)
from projects.pointnav_baselines.experiments.pointnav_habitat_mixin_ddppo import (
    PointNavHabitatMixInPPOConfig,
)
from projects.pointnav_baselines.experiments.pointnav_mixin_simpleconvgru import (
    PointNavMixInSimpleConvGRUConfig,
)


class PointNavHabitatRGBDDeterministiSimpleConvGRUDDPPOExperimentConfig(
    DebugPointNavHabitatBaseConfig,
    PointNavHabitatMixInPPOConfig,
    PointNavMixInSimpleConvGRUConfig,
):
    """An Point Navigation experiment configuration in Habitat with Depth
    input."""

    SENSORS = [
        RGBSensorHabitat(
            height=DebugPointNavHabitatBaseConfig.SCREEN_SIZE,
            width=DebugPointNavHabitatBaseConfig.SCREEN_SIZE,
            use_resnet_normalization=True,
        ),
        DepthSensorHabitat(
            height=DebugPointNavHabitatBaseConfig.SCREEN_SIZE,
            width=DebugPointNavHabitatBaseConfig.SCREEN_SIZE,
            use_normalization=True,
        ),
        TargetCoordinatesSensorHabitat(coordinate_dims=2),
    ]

    @classmethod
    def tag(cls):
        return "Debug-Pointnav-Habitat-RGBD-SimpleConv-DDPPO"
