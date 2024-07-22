import platform
from abc import ABC
from math import ceil
from typing import Dict, Any, List, Optional, Sequence

import gym
import numpy as np
import torch

from allenact.base_abstractions.experiment_config import MachineParams
from allenact.base_abstractions.preprocessor import SensorPreprocessorGraph
from allenact.base_abstractions.sensor import SensorSuite, ExpertActionSensor
from allenact.base_abstractions.task import TaskSampler
from allenact.utils.experiment_utils import evenly_distribute_count_into_bins
from allenact_plugins.manipulathor_plugin.manipulathor_constants import ENV_ARGS
from allenact_plugins.manipulathor_plugin.manipulathor_task_samplers import (
    SimpleArmPointNavGeneralSampler,
)
from allenact_plugins.manipulathor_plugin.manipulathor_viz import (
    ImageVisualizer,
    TestMetricLogger,
)
from projects.manipulathor_baselines.armpointnav_baselines.experiments.armpointnav_base import (
    ArmPointNavBaseConfig,
)


class ArmPointNavThorBaseConfig(ArmPointNavBaseConfig, ABC):
    """The base config for all iTHOR PointNav experiments."""

    TASK_SAMPLER = SimpleArmPointNavGeneralSampler
    VISUALIZE = False
    if platform.system() == "Darwin":
        VISUALIZE = True

    NUM_PROCESSES: Optional[int] = None
    TRAIN_GPU_IDS = list(range(torch.cuda.device_count()))
    SAMPLER_GPU_IDS = TRAIN_GPU_IDS
    VALID_GPU_IDS = [torch.cuda.device_count() - 1]
    TEST_GPU_IDS = [torch.cuda.device_count() - 1]

    TRAIN_DATASET_DIR: Optional[str] = None
    VAL_DATASET_DIR: Optional[str] = None

    CAP_TRAINING = None

    TRAIN_SCENES: Optional[List[str]] = None
    VAL_SCENES: Optional[List[str]] = None
    TEST_SCENES: Optional[List[str]] = None

    OBJECT_TYPES: Optional[Sequence[str]] = None
    VALID_SAMPLES_IN_SCENE = 1
    TEST_SAMPLES_IN_SCENE = 1

    NUMBER_OF_TEST_PROCESS = 10

    def __init__(self):
        super().__init__()

        assert (
            self.CAMERA_WIDTH == 224
            and self.CAMERA_HEIGHT == 224
            and self.VISIBILITY_DISTANCE == 1
            and self.STEP_SIZE == 0.25
        )
        self.ENV_ARGS = ENV_ARGS

    def machine_params(self, mode="train", **kwargs):
        sampler_devices: Sequence[int] = []
        if mode == "train":
            workers_per_device = 1
            gpu_ids = (
                []
                if not torch.cuda.is_available()
                else self.TRAIN_GPU_IDS * workers_per_device
            )
            nprocesses = (
                1
                if not torch.cuda.is_available()
                else evenly_distribute_count_into_bins(self.NUM_PROCESSES, len(gpu_ids))
            )
            sampler_devices = self.SAMPLER_GPU_IDS
        elif mode == "valid":
            nprocesses = 1
            gpu_ids = [] if not torch.cuda.is_available() else self.VALID_GPU_IDS
        elif mode == "test":
            nprocesses = self.NUMBER_OF_TEST_PROCESS if torch.cuda.is_available() else 1
            gpu_ids = [] if not torch.cuda.is_available() else self.TEST_GPU_IDS
        else:
            raise NotImplementedError("mode must be 'train', 'valid', or 'test'.")

        sensors = [*self.SENSORS]
        if mode != "train":
            sensors = [s for s in sensors if not isinstance(s, ExpertActionSensor)]

        sensor_preprocessor_graph = (
            SensorPreprocessorGraph(
                source_observation_spaces=SensorSuite(sensors).observation_spaces,
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
            sampler_devices=(
                sampler_devices if mode == "train" else gpu_ids
            ),  # ignored with > 1 gpu_ids
            sensor_preprocessor_graph=sensor_preprocessor_graph,
        )

    @classmethod
    def make_sampler_fn(cls, **kwargs) -> TaskSampler:
        from datetime import datetime

        now = datetime.now()
        exp_name_w_time = cls.__name__ + "_" + now.strftime("%m_%d_%Y_%H_%M_%S_%f")
        if cls.VISUALIZE:
            visualizers = [
                ImageVisualizer(exp_name=exp_name_w_time),
                TestMetricLogger(exp_name=exp_name_w_time),
            ]

            kwargs["visualizers"] = visualizers
        kwargs["objects"] = cls.OBJECT_TYPES
        kwargs["exp_name"] = exp_name_w_time
        return cls.TASK_SAMPLER(**kwargs)

    @staticmethod
    def _partition_inds(n: int, num_parts: int):
        return np.round(np.linspace(0, n, num_parts + 1, endpoint=True)).astype(
            np.int32
        )

    def _get_sampler_args_for_scene_split(
        self,
        scenes: List[str],
        process_ind: int,
        total_processes: int,
        seeds: Optional[List[int]] = None,
        deterministic_cudnn: bool = False,
    ) -> Dict[str, Any]:
        if total_processes > len(scenes):  # oversample some scenes -> bias
            if total_processes % len(scenes) != 0:
                print(
                    "Warning: oversampling some of the scenes to feed all processes."
                    " You can avoid this by setting a number of workers divisible by the number of scenes"
                )
            scenes = scenes * int(ceil(total_processes / len(scenes)))
            scenes = scenes[: total_processes * (len(scenes) // total_processes)]
        else:
            if len(scenes) % total_processes != 0:
                print(
                    "Warning: oversampling some of the scenes to feed all processes."
                    " You can avoid this by setting a number of workers divisor of the number of scenes"
                )
        inds = self._partition_inds(len(scenes), total_processes)

        return {
            "scenes": scenes[inds[process_ind] : inds[process_ind + 1]],
            "env_args": self.ENV_ARGS,
            "max_steps": self.MAX_STEPS,
            "sensors": self.SENSORS,
            "action_space": gym.spaces.Discrete(
                len(self.TASK_SAMPLER._TASK_TYPE.class_action_names())
            ),
            "seed": seeds[process_ind] if seeds is not None else None,
            "deterministic_cudnn": deterministic_cudnn,
            "rewards_config": self.REWARD_CONFIG,
        }

    def train_task_sampler_args(
        self,
        process_ind: int,
        total_processes: int,
        devices: Optional[List[int]] = None,
        seeds: Optional[List[int]] = None,
        deterministic_cudnn: bool = False,
    ) -> Dict[str, Any]:
        res = self._get_sampler_args_for_scene_split(
            self.TRAIN_SCENES,
            process_ind,
            total_processes,
            seeds=seeds,
            deterministic_cudnn=deterministic_cudnn,
        )
        res["scene_period"] = "manual"
        res["sampler_mode"] = "train"
        res["cap_training"] = self.CAP_TRAINING
        res["env_args"] = {}
        res["env_args"].update(self.ENV_ARGS)
        res["env_args"]["x_display"] = (
            ("0.%d" % devices[process_ind % len(devices)]) if len(devices) > 0 else None
        )
        return res

    def valid_task_sampler_args(
        self,
        process_ind: int,
        total_processes: int,
        devices: Optional[List[int]],
        seeds: Optional[List[int]] = None,
        deterministic_cudnn: bool = False,
    ) -> Dict[str, Any]:
        res = self._get_sampler_args_for_scene_split(
            self.VALID_SCENES,
            process_ind,
            total_processes,
            seeds=seeds,
            deterministic_cudnn=deterministic_cudnn,
        )
        res["scene_period"] = self.VALID_SAMPLES_IN_SCENE
        res["sampler_mode"] = "val"
        res["cap_training"] = self.CAP_TRAINING
        res["max_tasks"] = self.VALID_SAMPLES_IN_SCENE * len(res["scenes"])
        res["env_args"] = {}
        res["env_args"].update(self.ENV_ARGS)
        res["env_args"]["x_display"] = (
            ("0.%d" % devices[process_ind % len(devices)]) if len(devices) > 0 else None
        )
        return res

    def test_task_sampler_args(
        self,
        process_ind: int,
        total_processes: int,
        devices: Optional[List[int]],
        seeds: Optional[List[int]] = None,
        deterministic_cudnn: bool = False,
    ) -> Dict[str, Any]:
        res = self._get_sampler_args_for_scene_split(
            self.TEST_SCENES,
            process_ind,
            total_processes,
            seeds=seeds,
            deterministic_cudnn=deterministic_cudnn,
        )
        res["scene_period"] = self.TEST_SAMPLES_IN_SCENE
        res["sampler_mode"] = "test"
        res["env_args"] = {}
        res["cap_training"] = self.CAP_TRAINING
        res["env_args"].update(self.ENV_ARGS)
        res["env_args"]["x_display"] = (
            ("0.%d" % devices[process_ind % len(devices)]) if len(devices) > 0 else None
        )
        return res
