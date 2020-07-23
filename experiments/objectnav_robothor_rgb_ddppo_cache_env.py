from typing import Dict, Any, List, Optional
import json
from math import ceil

import gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torchvision import models
import numpy as np
import glob

from onpolicy_sync.losses.ppo import PPOConfig
from models.object_nav_models import ResnetTensorObjectNavActorCritic
from onpolicy_sync.losses import PPO
from rl_base.experiment_config import ExperimentConfig
from rl_base.task import TaskSampler
from rl_base.preprocessor import ObservationSet
from rl_robothor.robothor_tasks import ObjectNavTask
from rl_robothor.robothor_task_samplers import ObjectNavTaskSampler, ObjectNavDatasetTaskSampler
from rl_robothor.robothor_environment import RoboThorCachedEnvironment
from rl_robothor.robothor_sensors import ResNetRGBSensorHabitatCache
from rl_ai2thor.ai2thor_sensors import GoalObjectTypeThorSensor
from utils.experiment_utils import Builder, PipelineStage, TrainingPipeline, LinearDecay


class ObjectNavRoboThorRGBPPOExperimentConfig(ExperimentConfig):
    """An Object Navigation experiment configuration in RoboThor"""

    MAX_STEPS = 500

    VALIDATION_SAMPLES_PER_SCENE = 10

    ADVANCE_SCENE_ROLLOUT_PERIOD = 10000000000000

    NUM_PROCESSES = 120  # TODO 2 for debugging

    TARGET_TYPES = sorted(
        [
            'AlarmClock',
            'Apple',
            'BaseballBat',
            'BasketBall',
            'Bowl',
            'GarbageCan',
            'HousePlant',
            'Laptop',
            'Mug',
            'SprayBottle',
            'Television',
            'Vase'
        ]
    )

    SENSORS = [
        ResNetRGBSensorHabitatCache(
            {
                "uuid": "rgb_resnet",
            }
        ),
        GoalObjectTypeThorSensor({
            "object_types": TARGET_TYPES,
        }),
    ]

    PREPROCESSORS = []

    OBSERVATIONS = [
        "rgb_resnet",
        "goal_object_type_ind",
    ]

    @classmethod
    def tag(cls):
        return "ObjectNavRobothorRGBPPOCachedEnvironment"

    @classmethod
    def training_pipeline(cls, **kwargs):
        ppo_steps = int(300000000)
        lr = 3e-4
        num_mini_batch = 1
        update_repeats = 3
        num_steps = 30
        save_interval = 5000000
        log_interval = 10000
        gamma = 0.99
        use_gae = True
        gae_lambda = 0.95
        max_grad_norm = 0.5
        return TrainingPipeline(
            save_interval=save_interval,
            log_interval=log_interval,
            optimizer_builder=Builder(optim.Adam, dict(lr=lr)),
            num_mini_batch=num_mini_batch,
            update_repeats=update_repeats,
            max_grad_norm=max_grad_norm,
            num_steps=num_steps,
            named_losses={"ppo_loss": Builder(PPO, kwargs={}, default=PPOConfig,)},
            gamma=gamma,
            use_gae=use_gae,
            gae_lambda=gae_lambda,
            advance_scene_rollout_period=cls.ADVANCE_SCENE_ROLLOUT_PERIOD,
            pipeline_stages=[
                PipelineStage(loss_names=["ppo_loss"], end_criterion=ppo_steps)
            ],
            lr_scheduler_builder=Builder(
                LambdaLR, {"lr_lambda": LinearDecay(steps=ppo_steps)}
            ),
        )

    def split_num_processes(self, ndevices):
        assert self.NUM_PROCESSES >= ndevices, "NUM_PROCESSES {} < ndevices".format(self.NUM_PROCESSES, ndevices)
        res = [0] * ndevices
        for it in range(self.NUM_PROCESSES):
            res[it % ndevices] += 1
        return res

    def machine_params(self, mode="train", **kwargs):
        if mode == "train":
            workers_per_device = 1
            # gpu_ids = [] if not torch.cuda.is_available() else [0, 1, 2, 3, 4, 5, 6, 7] * workers_per_device  # TODO vs4 only has 7 gpus
            gpu_ids = [] if not torch.cuda.is_available() else [0, 1, 2, 3, 4, 5, 6] * workers_per_device  # TODO vs4 only has 7 gpus
            nprocesses = 1 if not torch.cuda.is_available() else self.split_num_processes(len(gpu_ids))
            sampler_devices = [0, 1, 2, 3, 4, 5, 6]  # TODO vs4 only has 7 gpus (ignored with > 1 gpu_ids)
            render_video = False
        elif mode == "valid":
            nprocesses = 15  # TODO debugging (0)
            gpu_ids = [] if not torch.cuda.is_available() else [7]
            render_video = False
        elif mode == "test":
            nprocesses = 15
            gpu_ids = [] if not torch.cuda.is_available() else [7]
            render_video = False
        else:
            raise NotImplementedError("mode must be 'train', 'valid', or 'test'.")

        # Disable parallelization for validation process
        if mode == "valid":
            for prep in self.PREPROCESSORS:
                prep.kwargs["config"]["parallel"] = False

        observation_set = Builder(ObservationSet, kwargs=dict(
            source_ids=self.OBSERVATIONS, all_preprocessors=self.PREPROCESSORS, all_sensors=self.SENSORS
        )) if mode == 'train' or nprocesses > 0 else None

        return {
            "nprocesses": nprocesses,
            "gpu_ids": gpu_ids,
            "sampler_devices": sampler_devices if mode == "train" else gpu_ids,  # ignored with > 1 gpu_ids
            "observation_set": observation_set,
            "render_video": render_video,
        }

    @classmethod
    def create_model(cls, **kwargs) -> nn.Module:
        return ResnetTensorObjectNavActorCritic(
            action_space=gym.spaces.Discrete(len(ObjectNavTask.action_names())),
            observation_space=kwargs["observation_set"].observation_spaces,
            goal_sensor_uuid="goal_object_type_ind",
            rgb_resnet_preprocessor_uuid="rgb_resnet",
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
        scenes_dir: str,
        process_ind: int,
        total_processes: int,
        seeds: Optional[List[int]] = None,
        deterministic_cudnn: bool = False,
    ) -> Dict[str, Any]:
        path = scenes_dir + "*.json.gz" if scenes_dir[-1] == "/" else scenes_dir + "/*.json.gz"
        scenes = [scene.split("/")[-1].split(".")[0] for scene in glob.glob(path)]
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
            "scenes": scenes[inds[process_ind]:inds[process_ind + 1]],
            "object_types": self.TARGET_TYPES,
            "max_steps": self.MAX_STEPS,
            "sensors": self.SENSORS,
            "action_space": gym.spaces.Discrete(len(ObjectNavTask.action_names())),
            "seed": seeds[process_ind] if seeds is not None else None,
            "deterministic_cudnn": deterministic_cudnn,
            "rewards_config": {
                "step_penalty": -0.01,
                "goal_success_reward": 10.0,
                "failed_stop_reward": 0.0,
                "shaping_weight": 1.0,  # applied to the decrease in distance to target
            },
            "env_class": RoboThorCachedEnvironment
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
            "dataset/robothor/objectnav/train/episodes",
            process_ind,
            total_processes,
            seeds=seeds,
            deterministic_cudnn=deterministic_cudnn,
        )
        res["scene_directory"] = "dataset/robothor/objectnav/train"
        res["loop_dataset"] = True
        res["env_args"] = {}
        res["env_args"]["x_display"] = (
            ("0.%d" % devices[process_ind % len(devices)]) if devices is not None and len(devices) > 0 else None
        )
        res["env_args"]["env_root_dir"] = "dataset/robothor/objectnav/train/view_caches"
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
            "dataset/robothor/objectnav/val/episodes",
            process_ind,
            total_processes,
            seeds=seeds,
            deterministic_cudnn=deterministic_cudnn,
        )
        res["scene_directory"] = "dataset/robothor/objectnav/val"
        res["loop_dataset"] = False
        res["env_args"] = {}
        res["env_args"]["x_display"] = (
            ("0.%d" % devices[process_ind % len(devices)]) if devices is not None and len(devices) > 0 else None
        )
        res["env_args"]["env_root_dir"] = "dataset/robothor/objectnav/val/view_caches"
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
            "dataset/robothor/objectnav/val/episodes",
            process_ind,
            total_processes,
            seeds=seeds,
            deterministic_cudnn=deterministic_cudnn,
        )
        res["scene_directory"] = "dataset/robothor/objectnav/val"
        res["loop_dataset"] = False
        res["env_args"] = {}
        # res["env_args"]["x_display"] = (
        #     ("0.%d" % devices[process_ind % len(devices)])
        #     if devices is not None and len(devices) > 0
        #     else None
        # )
        res["env_args"]["x_display"] = "10.0"
        res["env_args"]["env_root_dir"] = "dataset/robothor/objectnav/val/view_caches"
        return res
