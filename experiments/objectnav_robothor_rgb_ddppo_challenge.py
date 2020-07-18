from math import ceil
from typing import Dict, Any, List, Optional

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torchvision import models

from models.resnet_tensor_object_nav_models import ResnetTensorObjectNavActorCritic
from onpolicy_sync.losses import PPO
from onpolicy_sync.losses.ppo import PPOConfig
from rl_ai2thor.ai2thor_sensors import RGBSensorThor, GoalObjectTypeThorSensor
from rl_base.experiment_config import ExperimentConfig
from rl_base.preprocessor import ObservationSet
from rl_base.task import TaskSampler
from rl_habitat.habitat_preprocessors import ResnetPreProcessorHabitat
from rl_robothor.robothor_task_samplers import ObjectNavTaskSampler
from rl_robothor.robothor_tasks import ObjectNavTask
from utils.experiment_utils import Builder, PipelineStage, TrainingPipeline, LinearDecay


class ObjectNav_RoboThor_RGB_DDPPO_Challenge_ExperimentConfig(ExperimentConfig):
    """An Object Navigation experiment configuration in RoboThor."""

    TRAIN_SCENES = [
        "FloorPlan_Train%d_%d" % (wall + 1, furniture + 1)
        for wall in range(12)
        for furniture in range(5)
    ]

    VALID_SCENES = "rl_robothor/data/val.json"
    NUM_VALID_SCENES = 6116

    # VALID_SCENES = [
    #     "FloorPlan_Val%d_%d" % (wall + 1, furniture + 1)
    #     for wall in range(3)
    #     for furniture in range(5)
    # ]

    CAMERA_WIDTH = 640
    CAMERA_HEIGHT = 480

    SCREEN_SIZE = 224

    MAX_STEPS = 500
    ADVANCE_SCENE_ROLLOUT_PERIOD = 1000000000000000  # if more than 1 scene per worker

    NUM_PROCESSES = 60  # TODO 2 for debugging

    TARGET_TYPES = sorted(
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

    SENSORS = [
        RGBSensorThor(
            {
                "height": SCREEN_SIZE,
                "width": SCREEN_SIZE,
                "use_resnet_normalization": True,
                "uuid": "rgb_lowres",
            }
        ),
        GoalObjectTypeThorSensor({"object_types": TARGET_TYPES,}),
    ]

    PREPROCESSORS = [
        Builder(
            ResnetPreProcessorHabitat,
            dict(
                config={
                    "input_height": SCREEN_SIZE,
                    "input_width": SCREEN_SIZE,
                    "output_width": 7,
                    "output_height": 7,
                    "output_dims": 512,
                    "pool": False,
                    "torchvision_resnet_model": models.resnet18,
                    "input_uuids": ["rgb_lowres"],
                    "output_uuid": "rgb_resnet",
                    "parallel": False,  # TODO False for debugging or dd-ppo
                }
            ),
        ),
    ]

    OBSERVATIONS = [
        "rgb_resnet",
        "goal_object_type_ind",
    ]

    ENV_ARGS = dict(
        width=CAMERA_WIDTH,
        height=CAMERA_HEIGHT,
        continuousMode=True,
        agentType="stochastic",
        rotateStepDegrees=30.0,
        visibilityDistance=1.0,
        gridSize=0.25,
        snapToGrid=False,
        agentMode="bot",
    )

    @classmethod
    def tag(cls):
        return "ObjectNavRobothorRGBDDPPO_Challenge"

    @classmethod
    def training_pipeline(cls, **kwargs):
        ppo_steps = int(3e8)
        lr = 3e-4
        num_mini_batch = 1
        update_repeats = 3
        num_steps = 30
        save_interval = 200000
        log_interval = 1
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
        assert self.NUM_PROCESSES >= ndevices, "NUM_PROCESSES {} < ndevices {}".format(
            self.NUM_PROCESSES, ndevices
        )
        res = [0] * ndevices
        for it in range(self.NUM_PROCESSES):
            res[it % ndevices] += 1
        return res

    def machine_params(self, mode="train", **kwargs):
        if mode == "train":
            gpu_ids = (
                [] if not torch.cuda.is_available() else [0, 1, 2, 3, 4, 5, 6, 7]
            )  # TODO vs4 only has 7 gpus
            nprocesses = (
                1
                if not torch.cuda.is_available()
                else self.split_num_processes(len(gpu_ids))
            )
        elif mode == "valid":
            nprocesses = 1
            gpu_ids = (
                [] if not torch.cuda.is_available() else [7]
            )  # TODO vs4 only has 7 GPUs
        elif mode == "test":
            nprocesses = 1
            gpu_ids = (
                [] if not torch.cuda.is_available() else [0, 1, 2, 3, 4, 5, 6, 7]
            )  # TODO vs4 only has 7 gpus
        else:
            raise NotImplementedError("mode must be 'train', 'valid', or 'test'.")

        # Ensure preprocessor naive parallelization is disabled
        for prep in self.PREPROCESSORS:
            prep.kwargs["config"]["parallel"] = False

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

        return {
            "nprocesses": nprocesses,
            "gpu_ids": gpu_ids,
            "observation_set": observation_set,
            "visualizer": None,
        }

    @classmethod
    def create_model(cls, **kwargs) -> nn.Module:
        return ResnetTensorObjectNavActorCritic(
            action_space=gym.spaces.Discrete(len(ObjectNavTask.action_names())),
            observation_space=kwargs["observation_set"].observation_spaces,
            goal_sensor_uuid="goal_object_type_ind",
            resnet_preprocessor_uuid="rgb_resnet",
            rnn_hidden_size=512,
            goal_dims=32,
        )

    @classmethod
    def make_sampler_fn(cls, **kwargs) -> TaskSampler:
        return ObjectNavTaskSampler(**kwargs)

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
        res["env_args"] = {}
        res["env_args"].update(self.ENV_ARGS)
        res["env_args"]["x_display"] = (
            ("0.%d" % devices[process_ind % len(devices)])
            if devices is not None and len(devices) > 0
            else None
        )
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
        inds = self._partition_inds(self.NUM_VALID_SCENES, total_processes)
        res = dict(
            scenes=self.VALID_SCENES,  # special case: dataset file name (triggered by dataset_first, dataset_last >=0)
            object_types=self.TARGET_TYPES,
            max_steps=self.MAX_STEPS,
            sensors=self.SENSORS,
            action_space=gym.spaces.Discrete(len(ObjectNavTask.action_names())),
            seed=seeds[process_ind] if seeds is not None else None,
            deterministic_cudnn=deterministic_cudnn,
            dataset_first=inds[process_ind],
            dataset_last=inds[process_ind + 1] - 1,
            rewards_config={
                "step_penalty": -0.01,
                "goal_success_reward": 10.0,
                "failed_stop_reward": 0.0,
                "shaping_weight": 1.0,  # applied to the decrease in distance to target
            },
        )
        res["env_args"] = {}
        res["env_args"].update(self.ENV_ARGS)
        if isinstance(devices[0], int):
            res["env_args"]["x_display"] = (
                ("0.%d" % devices[process_ind % len(devices)])
                if devices is not None and len(devices) > 0
                else None
            )
        # else:
        #     print("Got devices {}".format(devices))
        return res
