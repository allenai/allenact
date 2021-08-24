from math import ceil
from typing import Dict, Any, List, Optional, Sequence, Union
import glob
import os
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torchvision import models

from allenact.base_abstractions.preprocessor import (
    Preprocessor,
    SensorPreprocessorGraph,
)
from allenact.embodiedai.preprocessors.resnet import ResNetPreprocessor
from allenact.embodiedai.sensors.vision_sensors import RGBSensor, DepthSensor
from allenact.embodiedai.preprocessors.resnet import ResNetPreprocessor
from allenact.utils.experiment_utils import Builder
from allenact.utils.experiment_utils import evenly_distribute_count_into_bins
from allenact.algorithms.onpolicy_sync.losses import PPO
from allenact.algorithms.onpolicy_sync.losses.ppo import PPOConfig
from allenact.base_abstractions.experiment_config import ExperimentConfig, MachineParams
from allenact.base_abstractions.sensor import SensorSuite
from allenact.base_abstractions.task import TaskSampler
from allenact.utils.experiment_utils import (
    Builder,
    PipelineStage,
    TrainingPipeline,
    LinearDecay,
)
from allenact_plugins.ithor_plugin.ithor_sensors import (
    RGBSensorThor,
    GoalObjectTypeThorSensor,
)
from allenact_plugins.ithor_plugin.ithor_task_samplers import (
    ObjectNavTaskSampler,
    ObjectNavDatasetTaskSampler,
)
from allenact_plugins.ithor_plugin.ithor_tasks import ObjectNaviThorGridTask
from projects.objectnav_baselines.models.object_nav_models import (
    ObjectNavBaselineActorCritic,
    ResnetTensorObjectNavActorCritic,
)


class ObjectNavThorPPOExperimentConfig(ExperimentConfig):
    """A simple object navigation experiment in THOR.

    Training with PPO.
    """

    # A simple setting, train/valid/test are all the same single scene
    # and we're looking for a single object
    OBJECT_TYPES = sorted(
        [
            "AlarmClock",
            "Apple",
            "Book",
            "Bowl",
            "Box",
            "Candle",
            "GarbageCan",
            "HousePlant",
            "Laptop",
            "SoapBottle",
            "Television",
            "Toaster",
        ]
    )
    train_path = os.path.join(
        os.getcwd(), "datasets/ithor-objectnav/train/episodes", "*.json.gz"
    )
    val_path = os.path.join(
        os.getcwd(), "datasets/ithor-objectnav/val/episodes", "*.json.gz"
    )
    test_path = os.path.join(
        os.getcwd(), "datasets/ithor-objectnav/val/episodes", "*.json.gz"
    )

    TRAIN_SCENES = [
        scene.split("/")[-1].split(".")[0] for scene in glob.glob(train_path)
    ]
    VALID_SCENES = [scene.split("/")[-1].split(".")[0] for scene in glob.glob(val_path)]
    TEST_SCENES = [scene.split("/")[-1].split(".")[0] for scene in glob.glob(test_path)]

    # Setting up sensors and basic environment details
    CAMERA_WIDTH = 400
    CAMERA_HEIGHT = 300
    SCREEN_SIZE = 224
    SENSORS = [
        RGBSensorThor(
            height=SCREEN_SIZE,
            width=SCREEN_SIZE,
            use_resnet_normalization=True,
            uuid="rgb_lowres",
        ),
        GoalObjectTypeThorSensor(object_types=OBJECT_TYPES),
    ]

    PREPROCESSORS = [
        Builder(
            ResNetPreprocessor,
            {
                "input_height": SCREEN_SIZE,
                "input_width": SCREEN_SIZE,
                "output_width": 7,
                "output_height": 7,
                "output_dims": 512,
                "pool": False,
                "torchvision_resnet_model": models.resnet18,
                "input_uuids": ["rgb_lowres"],
                "output_uuid": "rgb_resnet",
            },
        ),
    ]

    ENV_ARGS = {
        "player_screen_height": CAMERA_WIDTH,
        "player_screen_width": CAMERA_HEIGHT,
        "quality": "Very Low",
        "rotate_step_degrees": 30,
        "visibility_distance": 1.0,
        "grid_size": 0.25,
        "snap_to_grid": False
    }

    MAX_STEPS = 128
    ADVANCE_SCENE_ROLLOUT_PERIOD: Optional[int] = None
    VALID_SAMPLES_IN_SCENE = 10
    TEST_SAMPLES_IN_SCENE = 100

    DEFAULT_TRAIN_GPU_IDS = tuple(range(torch.cuda.device_count()))
    DEFAULT_VALID_GPU_IDS = (torch.cuda.device_count() - 1,)
    DEFAULT_TEST_GPU_IDS = (torch.cuda.device_count() - 1,)

    @classmethod
    def tag(cls):
        return "ObjectNavThorPPOResnetGRU"

    @classmethod
    def training_pipeline(cls, **kwargs):
        ppo_steps = int(300000000)
        lr = 3e-4
        num_mini_batch = 1 if not torch.cuda.is_available() else 6
        update_repeats = 4
        num_steps = 128
        metric_accumulate_interval = 10000  # Log every 10 max length tasks
        save_interval = 5000000
        gamma = 0.99
        use_gae = True
        gae_lambda = 0.95
        max_grad_norm = 0.5

        return TrainingPipeline(
            save_interval=save_interval,
            metric_accumulate_interval=metric_accumulate_interval,
            optimizer_builder=Builder(optim.Adam, dict(lr=lr)),
            num_mini_batch=num_mini_batch,
            update_repeats=update_repeats,
            max_grad_norm=max_grad_norm,
            num_steps=num_steps,
            named_losses={
                "ppo_loss": PPO(clip_decay=LinearDecay(ppo_steps), **PPOConfig),
            },
            gamma=gamma,
            use_gae=use_gae,
            gae_lambda=gae_lambda,
            advance_scene_rollout_period=cls.ADVANCE_SCENE_ROLLOUT_PERIOD,
            pipeline_stages=[
                PipelineStage(
                    loss_names=["ppo_loss"],
                    max_stage_steps=ppo_steps,
                ),
            ],
            lr_scheduler_builder=Builder(
                LambdaLR, {"lr_lambda": LinearDecay(steps=ppo_steps)}
            ),
        )

    @classmethod
    def machine_params(cls, mode="train", **kwargs):
        num_gpus = torch.cuda.device_count()
        has_gpu = num_gpus != 0

        if mode == "train":
            nprocesses = 40
            workers_per_device = 1
            gpu_ids = (
                [torch.device("cpu")]
                if not torch.cuda.is_available()
                else cls.DEFAULT_TRAIN_GPU_IDS * workers_per_device
            )
            nprocesses = evenly_distribute_count_into_bins(
                nprocesses, max(len(gpu_ids), 1)
            )
            sampler_devices = cls.DEFAULT_TRAIN_GPU_IDS
        elif mode == "valid":
            nprocesses = 1
            gpu_ids = [] if not torch.cuda.is_available() else cls.DEFAULT_VALID_GPU_IDS
        elif mode == "test":
            nprocesses = 1
            gpu_ids = [] if not torch.cuda.is_available() else cls.DEFAULT_TEST_GPU_IDS
        else:
            raise NotImplementedError("mode must be 'train', 'valid', or 'test'.")

        sensor_preprocessor_graph = (
            SensorPreprocessorGraph(
                source_observation_spaces=SensorSuite(cls.SENSORS).observation_spaces,
                preprocessors=cls.PREPROCESSORS,
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
    def create_model(cls, **kwargs) -> nn.Module:
        has_rgb = any(isinstance(s, RGBSensor) for s in cls.SENSORS)
        has_depth = any(isinstance(s, DepthSensor) for s in cls.SENSORS)
        goal_sensor_uuid = next(
            (s.uuid for s in cls.SENSORS if isinstance(s, GoalObjectTypeThorSensor)),
            None,
        )

        return ResnetTensorObjectNavActorCritic(
            action_space=gym.spaces.Discrete(
                len(ObjectNaviThorGridTask.class_action_names())
            ),
            observation_space=kwargs["sensor_preprocessor_graph"].observation_spaces,
            goal_sensor_uuid=goal_sensor_uuid,
            rgb_resnet_preprocessor_uuid="rgb_resnet" if has_rgb else None,
            depth_resnet_preprocessor_uuid="depth_resnet" if has_depth else None,
            hidden_size=512,
            goal_dims=32,
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
            "object_types": self.OBJECT_TYPES,
            "env_args": self.ENV_ARGS,
            "max_steps": self.MAX_STEPS,
            "sensors": self.SENSORS,
            "action_space": gym.spaces.Discrete(
                len(ObjectNaviThorGridTask.class_action_names())
            ),
            "seed": seeds[process_ind] if seeds is not None else None,
            "deterministic_cudnn": deterministic_cudnn,
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
        res["scene_directory"] = os.path.join(
            os.getcwd(), "datasets/ithor-objectnav/train/episodes"
        )
        res["scene_period"] = "manual"
        res["env_args"] = {}
        res["env_args"].update(self.ENV_ARGS)
        res["env_args"]["x_display"] = (
            ("0.%d" % devices[process_ind % len(devices)])
            if devices is not None and len(devices) > 0
            else None
        )
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
            self.VALID_SCENES,
            process_ind,
            total_processes,
            seeds=seeds,
            deterministic_cudnn=deterministic_cudnn,
        )
        res["scene_directory"] = os.path.join(
            os.getcwd(), "datasets/ithor-objectnav/val/episodes"
        )
        res["scene_period"] = self.VALID_SAMPLES_IN_SCENE
        res["max_tasks"] = self.VALID_SAMPLES_IN_SCENE * len(res["scenes"])
        res["env_args"] = {}
        res["env_args"].update(self.ENV_ARGS)
        res["env_args"]["x_display"] = (
            ("0.%d" % devices[process_ind % len(devices)])
            if devices is not None and len(devices) > 0
            else None
        )
        return res

    def test_task_sampler_args(
        self,
        process_ind: int,
        total_processes: int,
        devices: Optional[List[int]] = None,
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
        res["scene_directory"] = os.path.join(
            os.getcwd(), "datasets/ithor-objectnav/val/episodes"
        )
        res["scene_period"] = self.TEST_SAMPLES_IN_SCENE
        res["max_tasks"] = self.TEST_SAMPLES_IN_SCENE * len(res["scenes"])
        res["env_args"] = {}
        res["env_args"].update(self.ENV_ARGS)
        res["env_args"]["x_display"] = (
            ("0.%d" % devices[process_ind % len(devices)])
            if devices is not None and len(devices) > 0
            else None
        )
        return res
