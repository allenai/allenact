import os
from abc import ABC
from typing import Dict, Any, List, Optional, Sequence, Union

import gym
import torch

# noinspection PyUnresolvedReferences
import habitat
from allenact.base_abstractions.experiment_config import MachineParams
from allenact.base_abstractions.preprocessor import (
    SensorPreprocessorGraph,
    Preprocessor,
)
from allenact.base_abstractions.sensor import SensorSuite
from allenact.base_abstractions.task import TaskSampler
from allenact.embodiedai.sensors.vision_sensors import RGBSensor, DepthSensor
from allenact.utils.experiment_utils import evenly_distribute_count_into_bins, Builder
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
    training: bool,
    num_episode_sample: int,
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

    if not training:
        config.SEED = 0
        config.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = False

    if num_episode_sample > 0:
        config.ENVIRONMENT.ITERATOR_OPTIONS.NUM_EPISODE_SAMPLE = num_episode_sample

    config.MODE = mode

    config.freeze()

    return config


class PointNavHabitatBaseConfig(PointNavBaseConfig, ABC):
    """The base config for all Habitat PointNav experiments."""

    # selected auxiliary uuids
    ## if comment all the keys, then it's vanilla DD-PPO
    AUXILIARY_UUIDS = [
        # InverseDynamicsLoss.UUID,
        # TemporalDistanceLoss.UUID,
        # CPCA1Loss.UUID,
        # CPCA4Loss.UUID,
        # CPCA8Loss.UUID,
        # CPCA16Loss.UUID,
    ]
    ADD_PREV_ACTIONS = False
    MULTIPLE_BELIEFS = False
    BELIEF_FUSION = (  # choose one
        None
        # AttentiveFusion
        # AverageFusion
        # SoftmaxFusion
    )

    FAILED_END_REWARD = -1.0

    TASK_DATA_DIR_TEMPLATE = os.path.join(
        HABITAT_DATASETS_DIR, "pointnav/gibson/v1/{}/{}.json.gz"
    )
    BASE_CONFIG_YAML_PATH = os.path.join(
        HABITAT_CONFIGS_DIR, "tasks/pointnav_gibson.yaml"
    )

    ACTION_SPACE = gym.spaces.Discrete(len(PointNavTask.class_action_names()))

    DEFAULT_NUM_TRAIN_PROCESSES = (
        5 * torch.cuda.device_count() if torch.cuda.is_available() else 1
    )
    DEFAULT_NUM_TEST_PROCESSES = 10

    DEFAULT_TRAIN_GPU_IDS = tuple(range(torch.cuda.device_count()))
    DEFAULT_VALID_GPU_IDS = [torch.cuda.device_count() - 1]
    DEFAULT_TEST_GPU_IDS = [torch.cuda.device_count() - 1]

    def __init__(
        self,
        debug: bool = False,
        num_train_processes: Optional[int] = None,
        num_test_processes: Optional[int] = None,
        test_on_validation: bool = False,
        run_valid: bool = True,
        train_gpu_ids: Optional[Sequence[int]] = None,
        val_gpu_ids: Optional[Sequence[int]] = None,
        test_gpu_ids: Optional[Sequence[int]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        def v_or_default(v, default):
            return v if v is not None else default

        self.num_train_processes = v_or_default(
            num_train_processes, self.DEFAULT_NUM_TRAIN_PROCESSES
        )
        self.num_test_processes = v_or_default(
            num_test_processes, (10 if torch.cuda.is_available() else 1)
        )
        self.test_on_validation = test_on_validation
        self.run_valid = run_valid
        self.train_gpu_ids = v_or_default(train_gpu_ids, self.DEFAULT_TRAIN_GPU_IDS)
        self.val_gpu_ids = v_or_default(
            val_gpu_ids, self.DEFAULT_VALID_GPU_IDS if run_valid else []
        )
        self.test_gpu_ids = v_or_default(test_gpu_ids, self.DEFAULT_TEST_GPU_IDS)

        def create_config(
            mode: str,
            scenes_path: str,
            num_processes: int,
            simulator_gpu_ids: Sequence[int],
            training: bool = True,
            num_episode_sample: int = -1,
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
                training=training,
                num_episode_sample=num_episode_sample,
            )

        self.TRAIN_CONFIG = create_config(
            mode="train",
            scenes_path=self.train_scenes_path(),
            num_processes=self.num_train_processes,
            simulator_gpu_ids=self.train_gpu_ids,
            training=True,
        )
        self.VALID_CONFIG = create_config(
            mode="validate",
            scenes_path=self.valid_scenes_path(),
            num_processes=1,
            simulator_gpu_ids=self.val_gpu_ids,
            training=False,
            num_episode_sample=200,
        )
        self.TEST_CONFIG = create_config(
            mode="validate",
            scenes_path=self.test_scenes_path(),
            num_processes=self.num_test_processes,
            simulator_gpu_ids=self.test_gpu_ids,
            training=False,
        )

        self.TRAIN_CONFIGS_PER_PROCESS = construct_env_configs(
            self.TRAIN_CONFIG, allow_scene_repeat=True
        )

        if debug:
            get_logger().warning("IN DEBUG MODE, WILL ONLY USE `Adrian` SCENE!!!")
            for config in self.TRAIN_CONFIGS_PER_PROCESS:
                config.defrost()
                config.DATASET.CONTENT_SCENES = ["Adrian"]
                config.freeze()

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

    def preprocessors(self) -> Sequence[Union[Preprocessor, Builder[Preprocessor]]]:
        return tuple()

    def machine_params(self, mode="train", **kwargs):
        has_gpus = torch.cuda.is_available()
        if not has_gpus:
            gpu_ids = []
            nprocesses = 1
        elif mode == "train":
            gpu_ids = self.train_gpu_ids
            nprocesses = self.num_train_processes
        elif mode == "valid":
            gpu_ids = self.val_gpu_ids
            nprocesses = 1 if self.run_valid else 0
        elif mode == "test":
            gpu_ids = self.test_gpu_ids
            nprocesses = self.num_test_processes
        else:
            raise NotImplementedError("mode must be 'train', 'valid', or 'test'.")

        if has_gpus:
            nprocesses = evenly_distribute_count_into_bins(nprocesses, len(gpu_ids))

        sensor_preprocessor_graph = (
            SensorPreprocessorGraph(
                source_observation_spaces=SensorSuite(self.SENSORS).observation_spaces,
                preprocessors=self.preprocessors(),
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
            "action_space": self.ACTION_SPACE,
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
