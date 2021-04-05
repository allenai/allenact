import glob
import os
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
from allenact.utils.system import get_logger
from allenact_plugins.ithor_plugin.ithor_util import (
    horizontal_to_vertical_fov,
    get_open_x_displays,
)
from allenact_plugins.robothor_plugin.robothor_sensors import DepthSensorThor
from allenact_plugins.robothor_plugin.robothor_task_samplers import (
    ObjectNavDatasetTaskSampler,
)
from allenact_plugins.robothor_plugin.robothor_tasks import ObjectNavTask
from projects.objectnav_baselines.experiments.objectnav_base import ObjectNavBaseConfig
import ai2thor
from packaging import version

if ai2thor.__version__ not in ["0.0.1", None] and version.parse(
    ai2thor.__version__
) < version.parse("2.7.2"):
    raise ImportError(
        "To run the AI2-THOR ObjectNav baseline experiments you must use"
        " ai2thor version 2.7.1 or higher."
    )


class ObjectNavThorBaseConfig(ObjectNavBaseConfig, ABC):
    """The base config for all iTHOR PointNav experiments."""

    NUM_PROCESSES: Optional[int] = None
    TRAIN_GPU_IDS = list(range(torch.cuda.device_count()))
    SAMPLER_GPU_IDS = TRAIN_GPU_IDS
    VALID_GPU_IDS = [torch.cuda.device_count() - 1]
    TEST_GPU_IDS = [torch.cuda.device_count() - 1]

    TRAIN_DATASET_DIR: Optional[str] = None
    VAL_DATASET_DIR: Optional[str] = None
    TEST_DATASET_DIR: Optional[str] = None

    TARGET_TYPES: Optional[Sequence[str]] = None

    THOR_COMMIT_ID: Optional[str] = None

    @classmethod
    def env_args(cls):
        assert cls.THOR_COMMIT_ID is not None

        return dict(
            width=cls.CAMERA_WIDTH,
            height=cls.CAMERA_HEIGHT,
            commit_id=cls.THOR_COMMIT_ID,
            continuousMode=True,
            applyActionNoise=cls.STOCHASTIC,
            agentType="stochastic",
            rotateStepDegrees=cls.ROTATION_DEGREES,
            visibilityDistance=cls.VISIBILITY_DISTANCE,
            gridSize=cls.STEP_SIZE,
            snapToGrid=False,
            agentMode="locobot",
            fieldOfView=horizontal_to_vertical_fov(
                horizontal_fov_in_degrees=cls.HORIZONTAL_FIELD_OF_VIEW,
                width=cls.CAMERA_WIDTH,
                height=cls.CAMERA_HEIGHT,
            ),
            include_private_scenes=False,
            renderDepthImage=any(isinstance(s, DepthSensorThor) for s in cls.SENSORS),
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
        allow_oversample: bool = False,
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
            if not allow_oversample:
                raise RuntimeError(
                    f"Cannot have `total_processes > len(scenes)`"
                    f" ({total_processes} > {len(scenes)}) when `allow_oversample` is `False`."
                )

            if total_processes % len(scenes) != 0:
                get_logger().warning(oversample_warning)
            scenes = scenes * int(ceil(total_processes / len(scenes)))
            scenes = scenes[: total_processes * (len(scenes) // total_processes)]
        elif len(scenes) % total_processes != 0:
            get_logger().warning(oversample_warning)

        inds = self._partition_inds(len(scenes), total_processes)

        x_display: Optional[str] = None
        if platform.system() == "Linux":
            x_displays = get_open_x_displays(throw_error_if_empty=True)

            if len(devices) > len(x_displays):
                get_logger().warning(
                    f"More GPU devices found than X-displays (devices: `{x_displays}`, x_displays: `{x_displays}`)."
                    f" This is not necessarily a bad thing but may mean that you're not using GPU memory as"
                    f" efficiently as possible. Consider following the instructions here:"
                    f" https://allenact.org/installation/installation-framework/#installation-of-ithor-ithor-plugin"
                    f" describing how to start an X-display on every GPU."
                )
            x_display = x_displays[process_ind % len(x_displays)]

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
            "env_args": {**self.env_args(), "x_display": x_display,},
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
            scenes_dir=os.path.join(self.TRAIN_DATASET_DIR, "episodes"),
            process_ind=process_ind,
            total_processes=total_processes,
            devices=devices,
            seeds=seeds,
            deterministic_cudnn=deterministic_cudnn,
            allow_oversample=True,
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
            scenes_dir=os.path.join(self.VAL_DATASET_DIR, "episodes"),
            process_ind=process_ind,
            total_processes=total_processes,
            devices=devices,
            seeds=seeds,
            deterministic_cudnn=deterministic_cudnn,
            include_expert_sensor=False,
            allow_oversample=False,
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
        if self.TEST_DATASET_DIR is None:
            get_logger().warning(
                "No test dataset dir detected, running test on validation set instead."
            )
            return self.valid_task_sampler_args(
                process_ind=process_ind,
                total_processes=total_processes,
                devices=devices,
                seeds=seeds,
                deterministic_cudnn=deterministic_cudnn,
            )

        else:
            res = self._get_sampler_args_for_scene_split(
                scenes_dir=os.path.join(self.TEST_DATASET_DIR, "episodes"),
                process_ind=process_ind,
                total_processes=total_processes,
                devices=devices,
                seeds=seeds,
                deterministic_cudnn=deterministic_cudnn,
                include_expert_sensor=False,
                allow_oversample=False,
            )
            res["env_args"]["all_metadata_available"] = False
            res["rewards_config"] = {**res["rewards_config"], "shaping_weight": 0}
            res["scene_directory"] = self.TEST_DATASET_DIR
            res["loop_dataset"] = False
            return res
