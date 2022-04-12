import glob
import math
import os
from abc import ABC
from typing import Dict, Any, List, Optional, Sequence, Union

import gym
import numpy as np
import torch
from torch.distributions.utils import lazy_property

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
from allenact_plugins.habitat_plugin.habitat_task_samplers import ObjectNavTaskSampler
from allenact_plugins.habitat_plugin.habitat_tasks import ObjectNavTask
from allenact_plugins.habitat_plugin.habitat_utils import (
    get_habitat_config,
    construct_env_configs,
)
from projects.objectnav_baselines.experiments.objectnav_base import ObjectNavBaseConfig


def create_objectnav_config(
    config_yaml_path: str,
    mode: str,
    scenes_path: str,
    simulator_gpu_ids: Sequence[int],
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
    config.SIMULATOR.SEMANTIC_SENSOR.WIDTH = camera_width
    config.SIMULATOR.SEMANTIC_SENSOR.HEIGHT = camera_height

    assert rotation_degrees == config.SIMULATOR.TURN_ANGLE
    assert step_size == config.SIMULATOR.FORWARD_STEP_SIZE
    assert max_steps == config.ENVIRONMENT.MAX_EPISODE_STEPS
    config.SIMULATOR.MAX_EPISODE_STEPS = max_steps

    assert config.TASK.TYPE == "ObjectNav-v1"

    assert config.TASK.SUCCESS.SUCCESS_DISTANCE == 0.1
    assert config.TASK.DISTANCE_TO_GOAL.DISTANCE_TO == "VIEW_POINTS"

    config.TASK.SENSORS = ["OBJECTGOAL_SENSOR", "COMPASS_SENSOR", "GPS_SENSOR"]
    config.TASK.GOAL_SENSOR_UUID = "objectgoal"
    config.TASK.MEASUREMENTS = ["DISTANCE_TO_GOAL", "SUCCESS", "SPL", "SOFT_SPL"]

    if not training:
        config.SEED = 0
        config.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = False

    if num_episode_sample > 0:
        config.ENVIRONMENT.ITERATOR_OPTIONS.NUM_EPISODE_SAMPLE = num_episode_sample

    config.MODE = mode

    config.freeze()

    return config


class ObjectNavHabitatBaseConfig(ObjectNavBaseConfig, ABC):
    """The base config for all Habitat ObjectNav experiments."""

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

    ACTION_SPACE = gym.spaces.Discrete(len(ObjectNavTask.class_action_names()))

    DEFAULT_NUM_TRAIN_PROCESSES = (
        5 * torch.cuda.device_count() if torch.cuda.is_available() else 1
    )
    DEFAULT_NUM_TEST_PROCESSES = 11

    DEFAULT_TRAIN_GPU_IDS = tuple(range(torch.cuda.device_count()))
    DEFAULT_VALID_GPU_IDS = [torch.cuda.device_count() - 1]
    DEFAULT_TEST_GPU_IDS = tuple(range(torch.cuda.device_count()))

    def __init__(
        self,
        scene_dataset: str,  # Should be "mp3d" or "hm3d"
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

        self.scene_dataset = scene_dataset
        self.debug = debug

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

    def _create_config(
        self,
        mode: str,
        scenes_path: str,
        num_processes: int,
        simulator_gpu_ids: Sequence[int],
        training: bool = True,
        num_episode_sample: int = -1,
    ):
        return create_objectnav_config(
            config_yaml_path=self.BASE_CONFIG_YAML_PATH,
            mode=mode,
            scenes_path=scenes_path,
            simulator_gpu_ids=simulator_gpu_ids,
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

    @lazy_property
    def DEFAULT_OBJECT_CATEGORIES_TO_IND(self):
        if self.scene_dataset == "mp3d":
            return {
                "chair": 0,
                "table": 1,
                "picture": 2,
                "cabinet": 3,
                "cushion": 4,
                "sofa": 5,
                "bed": 6,
                "chest_of_drawers": 7,
                "plant": 8,
                "sink": 9,
                "toilet": 10,
                "stool": 11,
                "towel": 12,
                "tv_monitor": 13,
                "shower": 14,
                "bathtub": 15,
                "counter": 16,
                "fireplace": 17,
                "gym_equipment": 18,
                "seating": 19,
                "clothes": 20,
            }
        elif self.scene_dataset == "hm3d":
            return {
                "chair": 0,
                "bed": 1,
                "plant": 2,
                "toilet": 3,
                "tv_monitor": 4,
                "sofa": 5,
            }
        else:
            raise NotImplementedError

    @lazy_property
    def TASK_DATA_DIR_TEMPLATE(self):
        return os.path.join(
            HABITAT_DATASETS_DIR, f"objectnav/{self.scene_dataset}/v1/{{}}/{{}}.json.gz"
        )

    @lazy_property
    def BASE_CONFIG_YAML_PATH(self):
        return os.path.join(
            HABITAT_CONFIGS_DIR, f"tasks/objectnav_{self.scene_dataset}.yaml"
        )

    @lazy_property
    def TRAIN_CONFIG(self):
        return self._create_config(
            mode="train",
            scenes_path=self.train_scenes_path(),
            num_processes=self.num_train_processes,
            simulator_gpu_ids=self.train_gpu_ids,
            training=True,
        )

    @lazy_property
    def VALID_CONFIG(self):
        return self._create_config(
            mode="validate",
            scenes_path=self.valid_scenes_path(),
            num_processes=1,
            simulator_gpu_ids=self.val_gpu_ids,
            training=False,
            num_episode_sample=200,
        )

    @lazy_property
    def TEST_CONFIG(self):
        return self._create_config(
            mode="validate",
            scenes_path=self.test_scenes_path(),
            num_processes=self.num_test_processes,
            simulator_gpu_ids=self.test_gpu_ids,
            training=False,
        )

    @lazy_property
    def TRAIN_CONFIGS_PER_PROCESS(self):
        configs = construct_env_configs(self.TRAIN_CONFIG, allow_scene_repeat=True)

        if len(self.train_gpu_ids) >= 2:
            scenes_dir = configs[0].DATASET.SCENES_DIR
            memory_use_per_config = []
            for config in configs:
                assert len(config.DATASET.CONTENT_SCENES) == 1
                scene_name = config.DATASET.CONTENT_SCENES[0]

                paths = glob.glob(
                    os.path.join(
                        scenes_dir, self.scene_dataset, "**", f"{scene_name}.*"
                    ),
                    recursive=True,
                )

                if self.scene_dataset == "mp3d":
                    assert len(paths) == 4
                else:
                    assert len(paths) == 2

                memory_use_per_config.append(sum(os.path.getsize(p) for p in paths))

            max_configs_per_device = math.ceil(len(configs) / len(self.train_gpu_ids))
            mem_per_device = np.array([0.0 for _ in range(len(self.train_gpu_ids))])
            configs_per_device = [[] for _ in range(len(mem_per_device))]
            for mem, config in sorted(
                list(zip(memory_use_per_config, configs)), key=lambda x: x[0]
            ):
                ind = int(np.argmin(mem_per_device))
                config.defrost()
                config.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = self.train_gpu_ids[ind]
                config.freeze()
                configs_per_device[ind].append(config)

                mem_per_device[ind] += mem
                if len(configs_per_device[ind]) >= max_configs_per_device:
                    mem_per_device[ind] = float("inf")

            configs_per_device.sort(key=lambda x: len(x))
            configs = sum(configs_per_device, [])

        if self.debug:
            get_logger().warning(
                "IN DEBUG MODE, WILL ONLY USE `1LXtFkjw3qL` SCENE IN MP3D OR `1S7LAXRdDqK` scene in HM3D!!!"
            )
            for config in configs:
                config.defrost()
                if self.scene_dataset == "mp3d":
                    config.DATASET.CONTENT_SCENES = ["1LXtFkjw3qL"]
                elif self.scene_dataset == "hm3d":
                    config.DATASET.CONTENT_SCENES = ["1S7LAXRdDqK"]
                else:
                    raise NotImplementedError
                config.freeze()
        return configs

    @lazy_property
    def TEST_CONFIG_PER_PROCESS(self):
        return construct_env_configs(self.TEST_CONFIG, allow_scene_repeat=False)

    def train_scenes_path(self):
        return self.TASK_DATA_DIR_TEMPLATE.format(*(["train"] * 2))

    def valid_scenes_path(self):
        return self.TASK_DATA_DIR_TEMPLATE.format(*(["val"] * 2))

    def test_scenes_path(self):
        get_logger().warning("Running tests on the validation set!")
        return self.TASK_DATA_DIR_TEMPLATE.format(*(["val"] * 2))
        # return self.TASK_DATA_DIR_TEMPLATE.format(*(["test"] * 2))

    def tag(self):
        return f"ObjectNav-Habitat-{self.scene_dataset.upper()}"

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
        return ObjectNavTaskSampler(
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
            "action_space": gym.spaces.Discrete(
                len(ObjectNavTask.class_action_names())
            ),
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
            "action_space": gym.spaces.Discrete(
                len(ObjectNavTask.class_action_names())
            ),
        }
