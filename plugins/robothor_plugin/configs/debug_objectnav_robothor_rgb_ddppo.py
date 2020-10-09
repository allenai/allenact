import os
from typing import Sequence

import gym
import torch
import torch.nn as nn

from constants import ABS_PATH_OF_TOP_LEVEL_DIR
from core.base_abstractions.preprocessor import ObservationSet
from core.base_abstractions.task import TaskSampler
from plugins.ithor_plugin.ithor_sensors import GoalObjectTypeThorSensor
from plugins.ithor_plugin.ithor_sensors import RGBSensorThor
from plugins.robothor_plugin.robothor_task_samplers import ObjectNavDatasetTaskSampler
from plugins.robothor_plugin.robothor_tasks import PointNavTask
from projects.objectnav_baselines.models.object_nav_models import (
    ResnetTensorObjectNavActorCritic,
)
from projects.tutorials.pointnav_robothor_rgb_ddppo import (
    PointNavRoboThorRGBPPOExperimentConfig,
)
from utils.experiment_utils import Builder


class ObjectNavRoboThorRGBPPOExperimentConfig(PointNavRoboThorRGBPPOExperimentConfig):
    """An Object Navigation experiment configuration in RoboThor."""

    TRAINING_GPUS = tuple(range(min(torch.cuda.device_count(), 2)))

    # Simulator Parameters
    CAMERA_WIDTH = 400
    CAMERA_HEIGHT = 300
    SCREEN_SIZE = 224

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
        GoalObjectTypeThorSensor(
            object_types=sorted(
                [
                    "AlarmClock",
                    "Apple",
                    "BaseballBat",
                    "BasketBall",
                    "Bowl",
                    "GarbageCan",
                    "HousePlant",
                    "Laptop",
                    "Mug",
                    "SprayBottle",
                    "Television",
                    "Vase",
                ]
            ),
            uuid="object_type",
        ),
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

    def machine_params(self, mode="train", **kwargs):
        sampler_devices: Sequence[int] = []
        if mode == "train":
            workers_per_device = 1
            gpu_ids = (
                []
                if not torch.cuda.is_available()
                else self.TRAINING_GPUS * workers_per_device
            )
            nprocesses = (
                2
                if not torch.cuda.is_available()
                else self.split_num_processes(len(gpu_ids))
            )
            sampler_devices = self.TRAINING_GPUS
            render_video = False
        elif mode == "valid":
            nprocesses = 1
            gpu_ids = [] if not torch.cuda.is_available() else self.VALIDATION_GPUS
            render_video = False
        elif mode == "test":
            nprocesses = 1
            gpu_ids = [] if not torch.cuda.is_available() else self.TESTING_GPUS
            render_video = False
        else:
            raise NotImplementedError("mode must be 'train', 'valid', or 'test'.")

        # Disable parallelization for validation process
        if mode == "valid":
            for prep in self.PREPROCESSORS:
                prep.kwargs["parallel"] = False

        observation_set = (
            Builder(
                ObservationSet,
                kwargs=dict(
                    source_ids=self.OBSERVATIONS,
                    all_preprocessors=self.PREPROCESSORS,
                    all_sensors=self.SENSORS,
                ),
            )
            if mode == "train" or nprocesses > 0
            else None
        )

        return {
            "nprocesses": nprocesses,
            "gpu_ids": gpu_ids,
            "sampler_devices": sampler_devices if mode == "train" else gpu_ids,
            "observation_set": observation_set,
            "render_video": render_video,
        }

    # Define Task Sampler
    @classmethod
    def make_sampler_fn(cls, **kwargs) -> TaskSampler:
        return ObjectNavDatasetTaskSampler(**kwargs)
