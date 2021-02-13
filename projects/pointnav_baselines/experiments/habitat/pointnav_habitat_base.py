import os
from abc import ABC
from typing import Dict, Any, List, Optional, Sequence

import gym
import habitat
import torch

from allenact.base_abstractions.experiment_config import MachineParams
from allenact.base_abstractions.preprocessor import SensorPreprocessorGraph
from allenact.base_abstractions.sensor import SensorSuite
from allenact.embodiedai.sensors.vision_sensors import RGBSensor, DepthSensor
from allenact.base_abstractions.task import TaskSampler
from allenact.utils.experiment_utils import evenly_distribute_count_into_bins
from allenact.utils.system import get_logger
from allenact_plugins.habitat_plugin.habitat_constants import (
    HABITAT_DATASETS_DIR,
    HABITAT_CONFIGS_DIR,
    HABITAT_SCENE_DATASETS_DIR,
)
from allenact_plugins.habitat_plugin.habitat_task_samplers import PointNavTaskSampler
from allenact_plugins.habitat_plugin.habitat_tasks import PointNavTask
from allenact_plugins.habitat_plugin.habitat_utils import (
    get_habitat_config,
    construct_env_configs,
)
from projects.pointnav_baselines.experiments.pointnav_base import PointNavBaseConfig


def create_pointnav_config(
    config_yaml_path: str,
    mode: str,
    scenes_path: str,
    simulator_gpu_ids: Sequence[int],
    distance_to_goal: float,
    rotation_degrees: float,
    step_size: float,
    max_steps: int,
    num_processes: int,
    camera_width: int,
    camera_height: int,
    using_rgb: bool,
    using_depth: bool,
) -> habitat.Config:
    config = get_habitat_config(config_yaml_path)

    config.defrost()
    config.NUM_PROCESSES = num_processes
    config.SIMULATOR_GPU_IDS = simulator_gpu_ids
    config.DATASET.SCENES_DIR = HABITAT_SCENE_DATASETS_DIR

    config.DATASET.DATA_PATH = scenes_path
    config.SIMULATOR.AGENT_0.SENSORS = []

    if using_rgb:
        config.SIMULATOR.AGENT_0.SENSORS.append("RGB_SENSOR")
    if using_depth:
        config.SIMULATOR.AGENT_0.SENSORS.append("DEPTH_SENSOR")

    config.SIMULATOR.RGB_SENSOR.WIDTH = camera_width
    config.SIMULATOR.RGB_SENSOR.HEIGHT = camera_height
    config.SIMULATOR.DEPTH_SENSOR.WIDTH = camera_width
    config.SIMULATOR.DEPTH_SENSOR.HEIGHT = camera_height
    config.SIMULATOR.TURN_ANGLE = rotation_degrees
    config.SIMULATOR.FORWARD_STEP_SIZE = step_size
    config.ENVIRONMENT.MAX_EPISODE_STEPS = max_steps

    config.TASK.TYPE = "Nav-v0"
    config.TASK.SUCCESS_DISTANCE = distance_to_goal
    config.TASK.SENSORS = ["POINTGOAL_WITH_GPS_COMPASS_SENSOR"]
    config.TASK.POINTGOAL_WITH_GPS_COMPASS_SENSOR.GOAL_FORMAT = "POLAR"
    config.TASK.POINTGOAL_WITH_GPS_COMPASS_SENSOR.DIMENSIONALITY = 2
    config.TASK.GOAL_SENSOR_UUID = "pointgoal_with_gps_compass"
    config.TASK.MEASUREMENTS = ["DISTANCE_TO_GOAL", "SUCCESS", "SPL"]
    config.TASK.SPL.TYPE = "SPL"
    config.TASK.SPL.SUCCESS_DISTANCE = distance_to_goal
    config.TASK.SUCCESS.SUCCESS_DISTANCE = distance_to_goal

    config.MODE = mode

    config.freeze()

    return config


class PointNavHabitatBaseConfig(PointNavBaseConfig, ABC):
    """The base config for all Habitat PointNav experiments."""

    FAILED_END_REWARD = -1.0

    TASK_DATA_DIR_TEMPLATE = os.path.join(
        HABITAT_DATASETS_DIR, "pointnav/gibson/v1/{}/{}.json.gz"
    )
    BASE_CONFIG_YAML_PATH = os.path.join(
        HABITAT_CONFIGS_DIR, "tasks/pointnav_gibson.yaml"
    )

    NUM_TRAIN_PROCESSES = max(5 * torch.cuda.device_count() - 1, 4)
    NUM_VAL_PROCESSES = 1
    NUM_TEST_PROCESSES = 10

    TRAINING_GPUS = list(range(torch.cuda.device_count()))
    VALIDATION_GPUS = [torch.cuda.device_count() - 1]
    TESTING_GPUS = [torch.cuda.device_count() - 1]

    def __init__(self):
        super().__init__()

        def create_config(
            mode: str,
            scenes_path: str,
            num_processes: int,
            simulator_gpu_ids: Sequence[int],
        ):
            return create_pointnav_config(
                config_yaml_path=self.BASE_CONFIG_YAML_PATH,
                mode=mode,
                scenes_path=scenes_path,
                simulator_gpu_ids=simulator_gpu_ids,
                distance_to_goal=self.DISTANCE_TO_GOAL,
                rotation_degrees=self.ROTATION_DEGREES,
                step_size=self.STEP_SIZE,
                max_steps=self.MAX_STEPS,
                num_processes=num_processes,
                camera_width=self.CAMERA_WIDTH,
                camera_height=self.CAMERA_HEIGHT,
                using_rgb=any(isinstance(s, RGBSensor) for s in self.SENSORS),
                using_depth=any(isinstance(s, DepthSensor) for s in self.SENSORS),
            )

        self.TRAIN_CONFIG = create_config(
            mode="train",
            scenes_path=self.train_scenes_path(),
            num_processes=self.NUM_TRAIN_PROCESSES,
            simulator_gpu_ids=self.TRAINING_GPUS,
        )
        self.VALID_CONFIG = create_config(
            mode="validate",
            scenes_path=self.valid_scenes_path(),
            num_processes=self.NUM_VAL_PROCESSES,
            simulator_gpu_ids=self.VALIDATION_GPUS,
        )
        self.TEST_CONFIG = create_config(
            mode="validate",
            scenes_path=self.test_scenes_path(),
            num_processes=self.NUM_TEST_PROCESSES,
            simulator_gpu_ids=self.TESTING_GPUS,
        )

        self.TRAIN_CONFIGS_PER_PROCESS = construct_env_configs(
            self.TRAIN_CONFIG, allow_scene_repeat=True
        )
        self.TEST_CONFIG_PER_PROCESS = construct_env_configs(
            self.TEST_CONFIG, allow_scene_repeat=False
        )

    def train_scenes_path(self):
        return self.TASK_DATA_DIR_TEMPLATE.format(*(["train"] * 2))

    def valid_scenes_path(self):
        return self.TASK_DATA_DIR_TEMPLATE.format(*(["val"] * 2))

    def test_scenes_path(self):
        get_logger().warning("Running tests on the validation set!")
        return self.TASK_DATA_DIR_TEMPLATE.format(*(["val"] * 2))
        # return self.TASK_DATA_DIR_TEMPLATE.format(*(["test"] * 2))

    @classmethod
    def tag(cls):
        return "PointNav"

    def machine_params(self, mode="train", **kwargs):
        has_gpus = torch.cuda.is_available()
        if not has_gpus:
            gpu_ids = []
            nprocesses = 1
        elif mode == "train":
            gpu_ids = self.TRAINING_GPUS
            nprocesses = self.NUM_TRAIN_PROCESSES
        elif mode == "valid":
            gpu_ids = self.VALIDATION_GPUS
            nprocesses = self.NUM_VAL_PROCESSES
        elif mode == "test":
            gpu_ids = self.TESTING_GPUS
            nprocesses = self.NUM_TEST_PROCESSES
        else:
            raise NotImplementedError("mode must be 'train', 'valid', or 'test'.")

        if not has_gpus:
            nprocesses = 1
        else:
            nprocesses = evenly_distribute_count_into_bins(nprocesses, len(gpu_ids))

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
            devices=gpu_ids,
            sensor_preprocessor_graph=sensor_preprocessor_graph,
        )

    @classmethod
    def make_sampler_fn(cls, **kwargs) -> TaskSampler:
        return PointNavTaskSampler(
            **{"failed_end_reward": cls.FAILED_END_REWARD, **kwargs}  # type: ignore
        )

    def train_task_sampler_args(
        self,
        process_ind: int,
        total_processes: int,
        devices: Optional[List[int]] = None,
        seeds: Optional[List[int]] = None,
        deterministic_cudnn: bool = False,
    ) -> Dict[str, Any]:
        config = self.TRAIN_CONFIGS_PER_PROCESS[process_ind]
        return {
            "env_config": config,
            "max_steps": self.MAX_STEPS,
            "sensors": self.SENSORS,
            "action_space": gym.spaces.Discrete(len(PointNavTask.class_action_names())),
            "distance_to_goal": self.DISTANCE_TO_GOAL,
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
        return {
            "env_config": self.VALID_CONFIG,
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
        config = self.TEST_CONFIG_PER_PROCESS[process_ind]
        return {
            "env_config": config,
            "max_steps": self.MAX_STEPS,
            "sensors": self.SENSORS,
            "action_space": gym.spaces.Discrete(len(PointNavTask.class_action_names())),
            "distance_to_goal": self.DISTANCE_TO_GOAL,
        }
