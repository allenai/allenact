import glob
import os
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
from allenact.utils.system import get_logger
from allenact_plugins.robothor_plugin.robothor_sensors import DepthSensorThor
from allenact_plugins.robothor_plugin.robothor_task_samplers import (
    ObjectNavDatasetTaskSampler,
)
from allenact_plugins.robothor_plugin.robothor_tasks import ObjectNavTask
from projects.objectnav_baselines.experiments.objectnav_base import ObjectNavBaseConfig


class ObjectNavThorBaseConfig(ObjectNavBaseConfig, ABC):
    """The base config for all iTHOR PointNav experiments."""

    NUM_PROCESSES: Optional[int] = None
    TRAIN_GPU_IDS = list(range(torch.cuda.device_count()))
    SAMPLER_GPU_IDS = TRAIN_GPU_IDS
    VALID_GPU_IDS = [torch.cuda.device_count() - 1]
    TEST_GPU_IDS = [torch.cuda.device_count() - 1]

    TRAIN_DATASET_DIR: Optional[str] = None
    VAL_DATASET_DIR: Optional[str] = None

    TARGET_TYPES: Optional[Sequence[str]] = None

    def __init__(self):
        super().__init__()

        self.ENV_ARGS = dict(
            width=self.CAMERA_WIDTH,
            height=self.CAMERA_HEIGHT,
            continuousMode=True,
            applyActionNoise=self.STOCHASTIC,
            agentType="stochastic",
            rotateStepDegrees=self.ROTATION_DEGREES,
            visibilityDistance=self.VISIBILITY_DISTANCE,
            gridSize=self.STEP_SIZE,
            snapToGrid=False,
            agentMode="bot",
            include_private_scenes=False,
            renderDepthImage=any(isinstance(s, DepthSensorThor) for s in self.SENSORS),
        )

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
            nprocesses = 10 if torch.cuda.is_available() else 1
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
            sampler_devices=sampler_devices
            if mode == "train"
            else gpu_ids,  # ignored with > 1 gpu_ids
            sensor_preprocessor_graph=sensor_preprocessor_graph,
        )

    @classmethod
    def make_sampler_fn(cls, **kwargs) -> TaskSampler:
        return ObjectNavDatasetTaskSampler(**kwargs)

    @staticmethod
    def _partition_inds(n: int, num_parts: int):
        return np.round(np.linspace(0, n, num_parts + 1, endpoint=True)).astype(
            np.int32
        )

    def _get_sampler_args_for_scene_split(
        self,
        scenes_dir: str,
        process_ind: int,
        total_processes: int,
        devices: Optional[List[int]],
        seeds: Optional[List[int]],
        deterministic_cudnn: bool,
        include_expert_sensor: bool = True,
    ) -> Dict[str, Any]:
        path = os.path.join(scenes_dir, "*.json.gz")
        scenes = [scene.split("/")[-1].split(".")[0] for scene in glob.glob(path)]
        if len(scenes) == 0:
            raise RuntimeError(
                (
                    "Could find no scene dataset information in directory {}."
                    " Are you sure you've downloaded them? "
                    " If not, see https://allenact.org/installation/download-datasets/ information"
                    " on how this can be done."
                ).format(scenes_dir)
            )

        oversample_warning = (
            f"Warning: oversampling some of the scenes ({scenes}) to feed all processes ({total_processes})."
            " You can avoid this by setting a number of workers divisible by the number of scenes"
        )
        if total_processes > len(scenes):  # oversample some scenes -> bias
            if total_processes % len(scenes) != 0:
                get_logger().warning(oversample_warning)
            scenes = scenes * int(ceil(total_processes / len(scenes)))
            scenes = scenes[: total_processes * (len(scenes) // total_processes)]
        elif len(scenes) % total_processes != 0:
            get_logger().warning(oversample_warning)

        inds = self._partition_inds(len(scenes), total_processes)

        return {
            "scenes": scenes[inds[process_ind] : inds[process_ind + 1]],
            "object_types": self.TARGET_TYPES,
            "max_steps": self.MAX_STEPS,
            "sensors": [
                s
                for s in self.SENSORS
                if (include_expert_sensor or not isinstance(s, ExpertActionSensor))
            ],
            "action_space": gym.spaces.Discrete(
                len(ObjectNavTask.class_action_names())
            ),
            "seed": seeds[process_ind] if seeds is not None else None,
            "deterministic_cudnn": deterministic_cudnn,
            "rewards_config": self.REWARD_CONFIG,
            "env_args": {
                **self.ENV_ARGS,
                "x_display": (
                    f"0.{devices[process_ind % len(devices)]}"
                    if devices is not None
                    and len(devices) > 0
                    and devices[process_ind % len(devices)] >= 0
                    else None
                ),
            },
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
            os.path.join(self.TRAIN_DATASET_DIR, "episodes"),
            process_ind,
            total_processes,
            devices=devices,
            seeds=seeds,
            deterministic_cudnn=deterministic_cudnn,
        )
        res["scene_directory"] = self.TRAIN_DATASET_DIR
        res["loop_dataset"] = True
        res["allow_flipping"] = True
        return res

    def valid_task_sampler_args(
        self,
        process_ind: int,
        total_processes: int,
        devices: Optional[List[int]] = None,
        seeds: Optional[List[int]] = None,
        deterministic_cudnn: bool = False,
    ) -> Dict[str, Any]:
        res = self._get_sampler_args_for_scene_split(
            os.path.join(self.VAL_DATASET_DIR, "episodes"),
            process_ind,
            total_processes,
            devices=devices,
            seeds=seeds,
            deterministic_cudnn=deterministic_cudnn,
            include_expert_sensor=False,
        )
        res["scene_directory"] = self.VAL_DATASET_DIR
        res["loop_dataset"] = False
        return res

    def test_task_sampler_args(
        self,
        process_ind: int,
        total_processes: int,
        devices: Optional[List[int]] = None,
        seeds: Optional[List[int]] = None,
        deterministic_cudnn: bool = False,
    ) -> Dict[str, Any]:
        return self.valid_task_sampler_args(
            process_ind=process_ind,
            total_processes=total_processes,
            devices=devices,
            seeds=seeds,
            deterministic_cudnn=deterministic_cudnn,
        )
