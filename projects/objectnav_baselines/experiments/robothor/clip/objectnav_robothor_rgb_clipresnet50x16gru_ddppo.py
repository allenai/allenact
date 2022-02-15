from allenact_plugins.clip_plugin.clip_preprocessors import ClipResNetPreprocessor
from allenact_plugins.ithor_plugin.ithor_sensors import (
    RGBSensorThor,
    GoalObjectTypeThorSensor,
)
from projects.objectnav_baselines.experiments.clip.objectnav_mixin_clipresnetgru import (
    ObjectNavMixInClipResNetGRUConfig,
)
from projects.objectnav_baselines.experiments.objectnav_mixin_ddppo import (
    ObjectNavMixInPPOConfig,
)
from projects.objectnav_baselines.experiments.robothor.objectnav_robothor_base import (
    ObjectNavRoboThorBaseConfig,
)


class ObjectNavRoboThorRGBPPOExperimentConfig(
    ObjectNavRoboThorBaseConfig,
    ObjectNavMixInPPOConfig,
    ObjectNavMixInClipResNetGRUConfig,
):
    """An Object Navigation experiment configuration in RoboThor with RGB
    input."""

    CLIP_MODEL_TYPE = "RN50x16"

    SENSORS = [
        RGBSensorThor(
            height=ObjectNavRoboThorBaseConfig.SCREEN_SIZE,
            width=ObjectNavRoboThorBaseConfig.SCREEN_SIZE,
            use_resnet_normalization=True,
            uuid="rgb_lowres",
            mean=ClipResNetPreprocessor.CLIP_RGB_MEANS,
            stdev=ClipResNetPreprocessor.CLIP_RGB_STDS,
        ),
        GoalObjectTypeThorSensor(
            object_types=ObjectNavRoboThorBaseConfig.TARGET_TYPES,
        ),
    ]

    @classmethod
    def tag(cls):
        return "Objectnav-RoboTHOR-RGB-ClipResNet50x16GRU-DDPPO"
