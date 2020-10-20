import os
from abc import ABC
from typing import Dict, Any, List, Optional

import gym
import habitat
import torch

from constants import ABS_PATH_OF_TOP_LEVEL_DIR
from core.base_abstractions.experiment_config import MachineParams
from core.base_abstractions.preprocessor import ObservationSet
from core.base_abstractions.task import TaskSampler
from plugins.habitat_plugin.habitat_task_samplers import PointNavTaskSampler
from plugins.habitat_plugin.habitat_tasks import PointNavTask
from projects.pointnav_baselines.experiments.pointnav_base import PointNavBaseConfig
from utils.experiment_utils import Builder


class DebugPointNavHabitatBaseConfig(PointNavBaseConfig, ABC):
    """The base config for all Habitat PointNav experiments."""

    ADVANCE_SCENE_ROLLOUT_PERIOD: Optional[int] = None

    def __init__(self):
        super().__init__()

        habitat_data_dir = os.path.join(ABS_PATH_OF_TOP_LEVEL_DIR, "datasets/habitat")
        task_data_dir_template = os.path.join(
            habitat_data_dir, "datasets/pointnav/habitat-test-scenes/v1/{}/{}.json.gz"
        )
        self.TRAIN_SCENES = task_data_dir_template.format(*(["train"] * 2))
        self.VALID_SCENES = task_data_dir_template.format(*(["val"] * 2))
        self.TEST_SCENES = task_data_dir_template.format(*(["test"] * 2))

        scene_data_dir = os.path.join(habitat_data_dir, "scene_datasets")
        config_data_dir = os.path.join(habitat_data_dir, "configs")

        self.NUM_PROCESSES = 8 if torch.cuda.is_available() else 1
        self.CONFIG = habitat.get_config(
            os.path.join(config_data_dir, "debug_habitat_pointnav.yaml")
        )
        self.CONFIG.defrost()
        self.CONFIG.NUM_PROCESSES = self.NUM_PROCESSES
        self.CONFIG.SIMULATOR_GPU_IDS = [torch.cuda.device_count() - 1]
        self.CONFIG.DATASET.SCENES_DIR = scene_data_dir
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
        self.CONFIG.TASK.MEASUREMENTS = ["DISTANCE_TO_GOAL", "SPL"]
        self.CONFIG.TASK.SPL.TYPE = "SPL"
        self.CONFIG.TASK.SPL.SUCCESS_DISTANCE = self.DISTANCE_TO_GOAL

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

    def split_num_processes(self, ndevices):
        assert self.NUM_PROCESSES >= ndevices, "NUM_PROCESSES {} < ndevices {}".format(
            self.NUM_PROCESSES, ndevices
        )
        res = [0] * ndevices
        for it in range(self.NUM_PROCESSES):
            res[it % ndevices] += 1
        return res

    def machine_params(self, mode="train", **kwargs):
        if mode == "train":
            devices = self.TRAIN_GPUS
            nprocesses = self.split_num_processes(len(devices))
        elif mode == "valid":
            nprocesses = 0
            devices = self.VALIDATION_GPUS
        elif mode == "test":
            nprocesses = 1
            devices = self.TESTING_GPUS
        else:
            raise NotImplementedError("mode must be 'train', 'valid', or 'test'.")

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

        return MachineParams(
            nprocesses=nprocesses, devices=devices, observation_set=observation_set
        )

    @classmethod
    def make_sampler_fn(cls, **kwargs) -> TaskSampler:
        return PointNavTaskSampler(**kwargs)

    @staticmethod
    def make_easy_dataset(dataset: habitat.Dataset) -> habitat.Dataset:
        episodes = [
            e
            for e in dataset.episodes
            if float(e.info["geodesic_distance"]) < 1.5 and "skokloster" in e.scene_id
        ]
        # episodes = [episodes[0]]
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
            "filter_dataset_func": DebugPointNavHabitatBaseConfig.make_easy_dataset,
        }

    def valid_task_sampler_args(
        self,
        process_ind: int,
        total_processes: int,
        devices: Optional[List[int]] = None,
        seeds: Optional[List[int]] = None,
        deterministic_cudnn: bool = False,
    ) -> Dict[str, Any]:
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
