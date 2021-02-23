from plugins.ithor_plugin.ithor_sensors import RGBSensorThor
from plugins.robothor_plugin.robothor_sensors import (
    DepthSensorThor,
    GPSCompassSensorRoboThor,
)
from projects.pointnav_baselines.experiments.ithor.pointnav_ithor_base import (
    PointNaviThorBaseConfig,
)

from projects.pointnav_baselines.experiments.orbslam_pointnav_mixin_simpleconvgru import (
    PointNavMixInSimpleConvGRUConfig,
)
from projects.pointnav_baselines.experiments.pointnav_thor_mixin_ddppo import (
    PointNavThorMixInPPOConfig,
)

from plugins.orbslam2_plugin.orbslam2_sensors import ORBSLAMCompassSensorRoboThor


class PointNaviThorRGBDPPOExperimentConfig(
    PointNaviThorBaseConfig,
    PointNavThorMixInPPOConfig,
    PointNavMixInSimpleConvGRUConfig,
):
    """An Point Navigation experiment configuration in iThor with RGBD
    input."""

    NUM_PROCESSES = 1

    STEP_SIZE = 0.25
    ROTATION_DEGREES = 5.0

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
        ORBSLAMCompassSensorRoboThor(
            RGBSensorThor(
                height=PointNaviThorBaseConfig.CAMERA_HEIGHT,
                width=PointNaviThorBaseConfig.CAMERA_WIDTH,
                uuid='orbslam_rgb'
            ),
            DepthSensorThor(
                height=PointNaviThorBaseConfig.CAMERA_HEIGHT,
                width=PointNaviThorBaseConfig.CAMERA_WIDTH,
                uuid='orbslam_depth'
            ),
            vocab_file='/app/ORB_SLAM2/Vocabulary/ORBvoc.bin',
            use_slam_viewer=False,
            mem_limit=2048,
            uuid="target_coordinates_ind"
        )
    ]

    @classmethod
    def tag(cls):
        return "Pointnav-iTHOR-RGBD-SimpleConv-DDPPO"
