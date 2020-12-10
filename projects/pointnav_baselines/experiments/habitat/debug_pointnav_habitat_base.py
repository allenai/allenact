import os
from abc import ABC
from typing import Dict, Any, List, Optional, Sequence

import gym
import habitat
import torch

from core.base_abstractions.experiment_config import MachineParams
from core.base_abstractions.preprocessor import SensorPreprocessorGraph
from core.base_abstractions.sensor import SensorSuite
from core.base_abstractions.task import TaskSampler
from plugins.habitat_plugin.habitat_constants import (
    HABITAT_DATASETS_DIR,
    HABITAT_CONFIGS_DIR,
    HABITAT_SCENE_DATASETS_DIR,
)
from plugins.habitat_plugin.habitat_task_samplers import PointNavTaskSampler
from plugins.habitat_plugin.habitat_tasks import PointNavTask
from plugins.habitat_plugin.habitat_utils import get_habitat_config
from projects.pointnav_baselines.experiments.pointnav_base import PointNavBaseConfig
from utils.experiment_utils import evenly_distribute_count_into_bins


class DebugPointNavHabitatBaseConfig(PointNavBaseConfig, ABC):
    """The base config for all Habitat PointNav experiments."""

    ADVANCE_SCENE_ROLLOUT_PERIOD: Optional[int] = None

    def __init__(self):
        super().__init__()

        task_data_dir_template = os.path.join(
            HABITAT_DATASETS_DIR, "pointnav/habitat-test-scenes/v1/{}/{}.json.gz"
        )
        self.TRAIN_SCENES = task_data_dir_template.format(*(["train"] * 2))
        self.VALID_SCENES = task_data_dir_template.format(*(["val"] * 2))
        self.TEST_SCENES = task_data_dir_template.format(*(["test"] * 2))

        self.NUM_PROCESSES = 8 if torch.cuda.is_available() else 4
        self.CONFIG = get_habitat_config(
            os.path.join(HABITAT_CONFIGS_DIR, "debug_habitat_pointnav.yaml")
        )
        self.CONFIG.defrost()
        self.CONFIG.NUM_PROCESSES = self.NUM_PROCESSES
        self.CONFIG.SIMULATOR_GPU_IDS = [torch.cuda.device_count() - 1]
        self.CONFIG.DATASET.SCENES_DIR = HABITAT_SCENE_DATASETS_DIR
        self.CONFIG.DATASET.POINTNAVV1.CONTENT_SCENES = ["*"]
        self.CONFIG.DATASET.DATA_PATH = self.TRAIN_SCENES
        self.CONFIG.SIMULATOR.AGENT_0.SENSORS = ["RGB_SENSOR"]
        self.CONFIG.SIMULATOR.RGB_SENSOR.WIDTH = self.CAMERA_WIDTH
        self.CONFIG.SIMULATOR.RGB_SENSOR.HEIGHT = self.CAMERA_HEIGHT
        self.CONFIG.SIMULATOR.DEPTH_SENSOR.WIDTH = self.CAMERA_WIDTH
        self.CONFIG.SIMULATOR.DEPTH_SENSOR.HEIGHT = self.CAMERA_HEIGHT
        self.CONFIG.SIMULATOR.TURN_ANGLE = self.ROTATION_DEGREES
        self.CONFIG.SIMULATOR.FORWARD_STEP_SIZE = self.STEP_SIZE
        self.CONFIG.ENVIRONMENT.MAX_EPISODE_STEPS = self.MAX_STEPS

        self.CONFIG.TASK.TYPE = "Nav-v0"
        self.CONFIG.TASK.SUCCESS_DISTANCE = self.DISTANCE_TO_GOAL
        self.CONFIG.TASK.SENSORS = ["POINTGOAL_WITH_GPS_COMPASS_SENSOR"]
        self.CONFIG.TASK.POINTGOAL_WITH_GPS_COMPASS_SENSOR.GOAL_FORMAT = "POLAR"
        self.CONFIG.TASK.POINTGOAL_WITH_GPS_COMPASS_SENSOR.DIMENSIONALITY = 2
        self.CONFIG.TASK.GOAL_SENSOR_UUID = "pointgoal_with_gps_compass"
        self.CONFIG.TASK.MEASUREMENTS = ["DISTANCE_TO_GOAL", "SUCCESS", "SPL"]
        self.CONFIG.TASK.SPL.TYPE = "SPL"
        self.CONFIG.TASK.SPL.SUCCESS_DISTANCE = self.DISTANCE_TO_GOAL
        self.CONFIG.TASK.SUCCESS.SUCCESS_DISTANCE = self.DISTANCE_TO_GOAL

        self.CONFIG.MODE = "train"

        self.TRAIN_GPUS = [torch.cuda.device_count() - 1]
        self.VALIDATION_GPUS = [torch.cuda.device_count() - 1]
        self.TESTING_GPUS = [torch.cuda.device_count() - 1]

        self.TRAIN_CONFIGS = None
        self.TEST_CONFIGS = None
        self.SENSORS = None

    @classmethod
    def tag(cls):
        return "PointNav"

    def machine_params(self, mode="train", **kwargs):
        if mode == "train":
            devices = self.TRAIN_GPUS
            nprocesses = evenly_distribute_count_into_bins(
                self.NUM_PROCESSES, len(devices)
            )
        elif mode == "valid":
            nprocesses = 0
            devices = self.VALIDATION_GPUS
        elif mode == "test":
            nprocesses = 1
            devices = self.TESTING_GPUS
        else:
            raise NotImplementedError("mode must be 'train', 'valid', or 'test'.")

        sensor_preprocessor_graph = (
            SensorPreprocessorGraph(
                source_observation_spaces=SensorSuite(self.SENSORS).observation_spaces,
                preprocessors=self.PREPROCESSORS,
            )
            if mode == "train"
            or (
                (isinstance(nprocesses, int) and nprocesses > 0)
                or (isinstance(nprocesses, Sequence) and sum(nprocesses) > 0)
            )
            else None
        )

        return MachineParams(
            nprocesses=nprocesses,
            devices=devices,
            sensor_preprocessor_graph=sensor_preprocessor_graph,
        )

    @classmethod
    def make_sampler_fn(cls, **kwargs) -> TaskSampler:
        return PointNavTaskSampler(**kwargs)

    @staticmethod
    def make_easy_dataset(dataset: habitat.Dataset) -> habitat.Dataset:
        episodes = [
            e for e in dataset.episodes if float(e.info["geodesic_distance"]) < 1.5
        ]
        for i, e in enumerate(episodes):
            e.episode_id = str(i)
        dataset.episodes = episodes
        return dataset

    def train_task_sampler_args(
        self,
        process_ind: int,
        total_processes: int,
        devices: Optional[List[int]] = None,
        seeds: Optional[List[int]] = None,
        deterministic_cudnn: bool = False,
    ) -> Dict[str, Any]:
        config = self.TRAIN_CONFIGS[process_ind]
        return {
            "env_config": config,
            "max_steps": self.MAX_STEPS,
            "sensors": self.SENSORS,
            "action_space": gym.spaces.Discrete(len(PointNavTask.class_action_names())),
            "distance_to_goal": self.DISTANCE_TO_GOAL,
            # "filter_dataset_func": DebugPointNavHabitatBaseConfig.make_easy_dataset,
        }

    def valid_task_sampler_args(
        self,
        process_ind: int,
        total_processes: int,
        devices: Optional[List[int]] = None,
        seeds: Optional[List[int]] = None,
        deterministic_cudnn: bool = False,
    ) -> Dict[str, Any]:
        if total_processes != 1:
            raise NotImplementedError(
                "In validation, `total_processes` must equal 1 for habitat tasks"
            )
        config = self.CONFIG.clone()
        config.defrost()
        config.DATASET.DATA_PATH = self.VALID_SCENES
        config.MODE = "validate"
        config.freeze()
        return {
            "env_config": config,
            "max_steps": self.MAX_STEPS,
            "sensors": self.SENSORS,
            "action_space": gym.spaces.Discrete(len(PointNavTask.class_action_names())),
            "distance_to_goal": self.DISTANCE_TO_GOAL,
        }

    def test_task_sampler_args(
        self,
        process_ind: int,
        total_processes: int,
        devices: Optional[List[int]] = None,
        seeds: Optional[List[int]] = None,
        deterministic_cudnn: bool = False,
    ) -> Dict[str, Any]:
        config = self.TEST_CONFIGS[process_ind]
        return {
            "env_config": config,
            "max_steps": self.MAX_STEPS,
            "sensors": self.SENSORS,
            "action_space": gym.spaces.Discrete(len(PointNavTask.class_action_names())),
            "distance_to_goal": self.DISTANCE_TO_GOAL,
        }
