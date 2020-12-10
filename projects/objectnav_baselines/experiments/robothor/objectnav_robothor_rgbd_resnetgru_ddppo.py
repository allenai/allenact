from torchvision import models

from core.base_abstractions.preprocessor import ResNetPreprocessor
from plugins.ithor_plugin.ithor_sensors import RGBSensorThor, GoalObjectTypeThorSensor
from plugins.robothor_plugin.robothor_sensors import DepthSensorThor
from projects.objectnav_baselines.experiments.robothor.objectnav_robothor_base import (
    ObjectNavRoboThorBaseConfig,
)
from utils.experiment_utils import Builder


class ObjectNavRoboThorRGBPPOExperimentConfig(ObjectNavRoboThorBaseConfig):
    """An Object Navigation experiment configuration in RoboThor with RGBD
    input."""

    SENSORS = [
        RGBSensorThor(
            height=ObjectNavRoboThorBaseConfig.SCREEN_SIZE,
            width=ObjectNavRoboThorBaseConfig.SCREEN_SIZE,
            use_resnet_normalization=True,
            uuid="rgb_lowres",
        ),
        DepthSensorThor(
            height=ObjectNavRoboThorBaseConfig.SCREEN_SIZE,
            width=ObjectNavRoboThorBaseConfig.SCREEN_SIZE,
            use_normalization=True,
            uuid="depth_lowres",
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
                "input_uuids": ["depth_lowres"],
                "output_uuid": "depth_resnet",
                "parallel": False,
            },
        ),
    ]

    @classmethod
    def tag(cls):
        return "Objectnav-RoboTHOR-RGBD-ResNetGRU-DDPPO"
