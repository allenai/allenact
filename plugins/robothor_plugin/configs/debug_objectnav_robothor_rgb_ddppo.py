import os

import gym
import torch.nn as nn

from constants import ABS_PATH_OF_TOP_LEVEL_DIR
from projects.tutorials.pointnav_robothor_rgb_ddppo import (
    PointNavRoboThorRGBPPOExperimentConfig,
)
from projects.objectnav_baselines.models.object_nav_models import (
    ResnetTensorObjectNavActorCritic,
)
from plugins.ithor_plugin.ithor_sensors import RGBSensorThor
from core.base_abstractions.task import TaskSampler
from plugins.ithor_plugin.ithor_sensors import GoalObjectTypeThorSensor
from plugins.robothor_plugin.robothor_task_samplers import ObjectNavDatasetTaskSampler
from plugins.robothor_plugin.robothor_tasks import PointNavTask


class ObjectNavRoboThorRGBPPOExperimentConfig(PointNavRoboThorRGBPPOExperimentConfig):
    """An Object Navigation experiment configuration in RoboThor."""

    # Dataset Parameters
    TRAIN_DATASET_DIR = os.path.join(
        ABS_PATH_OF_TOP_LEVEL_DIR, "datasets/robothor-objectnav/debug"
    )
    VAL_DATASET_DIR = os.path.join(
        ABS_PATH_OF_TOP_LEVEL_DIR, "datasets/robothor-objectnav/debug"
    )

    SENSORS = [
        RGBSensorThor(
            height=PointNavRoboThorRGBPPOExperimentConfig.SCREEN_SIZE,
            width=PointNavRoboThorRGBPPOExperimentConfig.SCREEN_SIZE,
            use_resnet_normalization=True,
            uuid="rgb_lowres",
        ),
        GoalObjectTypeThorSensor(object_types=["Television"], uuid="object_type"),
    ]

    OBSERVATIONS = [
        "rgb_resnet",
        "object_type",
    ]

    @classmethod
    def tag(cls):
        return "DebugObjectNavRobothorRGBPPO"

    @classmethod
    def training_pipeline(cls, **kwargs):
        pipeline = super().training_pipeline(**kwargs)
        pipeline.save_interval = 500000
        return pipeline

    # Define Model
    @classmethod
    def create_model(cls, **kwargs) -> nn.Module:
        return ResnetTensorObjectNavActorCritic(
            action_space=gym.spaces.Discrete(len(PointNavTask.class_action_names())),
            observation_space=kwargs["observation_set"].observation_spaces,
            goal_sensor_uuid="object_type",
            rgb_resnet_preprocessor_uuid="rgb_resnet",
            hidden_size=512,
            goal_dims=32,
        )

    # Define Task Sampler
    @classmethod
    def make_sampler_fn(cls, **kwargs) -> TaskSampler:
        return ObjectNavDatasetTaskSampler(**kwargs)
