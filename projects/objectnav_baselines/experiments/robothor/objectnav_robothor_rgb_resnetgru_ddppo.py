from torchvision import models

from core.base_abstractions.preprocessor import ResNetPreprocessor
from plugins.ithor_plugin.ithor_sensors import RGBSensorThor, GoalObjectTypeThorSensor
from projects.objectnav_baselines.experiments.robothor.objectnav_robothor_base import (
    ObjectNavRoboThorBaseConfig,
)
from projects.objectnav_baselines.experiments.robothor.objectnav_robothor_ddppo_base import (
    ObjectNavRoboThorPPOBaseExperimentConfig,
)
from projects.objectnav_baselines.experiments.robothor.objectnav_robothor_resnetgru_base import (
    ObjectNavRoboThorResNetGRUBaseExperimentConfig,
)
from utils.experiment_utils import Builder


class ObjectNaviThorRGBPPOExperimentConfig(
    ObjectNavRoboThorPPOBaseExperimentConfig,
    ObjectNavRoboThorResNetGRUBaseExperimentConfig,
):
    """An Object Navigation experiment configuration in RoboThor with RGB
    input."""

    SENSORS = [
        RGBSensorThor(
            height=ObjectNavRoboThorBaseConfig.SCREEN_SIZE,
            width=ObjectNavRoboThorBaseConfig.SCREEN_SIZE,
            use_resnet_normalization=True,
            uuid="rgb_lowres",
        ),
        GoalObjectTypeThorSensor(
            object_types=ObjectNavRoboThorBaseConfig.TARGET_TYPES,
        ),
    ]

    PREPROCESSORS = [
        Builder(
            ResNetPreprocessor,
            {
                "input_height": ObjectNavRoboThorBaseConfig.SCREEN_SIZE,
                "input_width": ObjectNavRoboThorBaseConfig.SCREEN_SIZE,
                "output_width": 7,
                "output_height": 7,
                "output_dims": 512,
                "pool": False,
                "torchvision_resnet_model": models.resnet18,
                "input_uuids": ["rgb_lowres"],
                "output_uuid": "rgb_resnet",
                "parallel": False,
            },
        ),
    ]

    @classmethod
    def tag(cls):
        return "Objectnav-RoboTHOR-RGB-ResNetGRU-DDPPO"
