import glob
import os
from abc import ABC
from math import ceil
from typing import Dict, Any, List, Optional, Sequence, Union

import gym
import numpy as np
import torch

from constants import ABS_PATH_OF_TOP_LEVEL_DIR
from core.base_abstractions.preprocessor import ObservationSet, Preprocessor
from core.base_abstractions.sensor import ExpertActionSensor, Sensor
from core.base_abstractions.task import TaskSampler
from plugins.robothor_plugin.robothor_sensors import DepthSensorThor
from plugins.robothor_plugin.robothor_task_samplers import ObjectNavDatasetTaskSampler
from plugins.robothor_plugin.robothor_tasks import ObjectNavTask
from projects.objectnav_baselines.experiments.objectnav_base import ObjectNavBaseConfig
from utils.experiment_utils import Builder
from utils.system import get_logger


class ObjectNavRoboThorBaseConfig(ObjectNavBaseConfig, ABC):
    """The base config for all RoboTHOR ObjectNav experiments."""

    ADVANCE_SCENE_ROLLOUT_PERIOD: Optional[int] = None
    SENSORS: Optional[Sequence[Sensor]] = None
    PREPROCESSORS: Sequence[Union[Preprocessor, Builder[Preprocessor]]] = tuple()

    NUM_PROCESSES = 60
    TRAIN_GPU_IDS = list(range(min(torch.cuda.device_count(), 8)))
    SAMPLER_GPU_IDS = TRAIN_GPU_IDS
    VALID_GPU_IDS = [torch.cuda.device_count() - 1] if torch.cuda.is_available() else []
    TEST_GPU_IDS = [0]

    TRAIN_DATASET_DIR = os.path.join(
        ABS_PATH_OF_TOP_LEVEL_DIR, "datasets/robothor-objectnav/train"
    )
    VAL_DATASET_DIR = os.path.join(
        ABS_PATH_OF_TOP_LEVEL_DIR, "datasets/robothor-objectnav/val"
    )

    TARGET_TYPES = tuple(
        sorted(
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
        )
    )

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

    @staticmethod
    def split_num_processes(nprocesses: int, ndevices: int):
        assert nprocesses >= ndevices, "NUM_PROCESSES {} < ndevices {}".format(
            nprocesses, ndevices
        )
        res = [0] * ndevices
        for it in range(nprocesses):
            res[it % ndevices] += 1
        return res

    def machine_params(self, mode="train", **kwargs):
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
                else self.split_num_processes(self.NUM_PROCESSES, ndevices=len(gpu_ids))
            )
            sampler_devices = self.SAMPLER_GPU_IDS
        elif mode == "valid":
            nprocesses = 1 if torch.cuda.is_available() else 1
            gpu_ids = [] if not torch.cuda.is_available() else self.VALID_GPU_IDS
        elif mode == "test":
            nprocesses = 15 if torch.cuda.is_available() else 1
            gpu_ids = [] if not torch.cuda.is_available() else self.TEST_GPU_IDS
        else:
            raise NotImplementedError("mode must be 'train', 'valid', or 'test'.")

        sensors = [*self.SENSORS]
        if mode != "train":
            sensors = [s for s in sensors if not isinstance(s, ExpertActionSensor)]

        # Disable parallelization for validation process
        if mode == "valid":
            for prep in self.PREPROCESSORS:
                prep.kwargs["parallel"] = False

        observation_set = (
            Builder(
                ObservationSet,
                kwargs=dict(
                    source_ids=[s.uuid for s in sensors]
                    + [
                        (p() if isinstance(p, Builder) else p).uuid
                        for p in self.PREPROCESSORS
                    ],
                    all_preprocessors=self.PREPROCESSORS,
                    all_sensors=sensors,
                ),
            )
            if mode == "train"
            or (
                (isinstance(nprocesses, int) and nprocesses > 0)
                or (isinstance(nprocesses, Sequence) and sum(nprocesses) > 0)
            )
            else None
        )

        return {
            "nprocesses": nprocesses,
            "gpu_ids": gpu_ids,
            "sampler_devices": sampler_devices
            if mode == "train"
            else gpu_ids,  # ignored with > 1 gpu_ids
            "observation_set": observation_set,
        }

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
