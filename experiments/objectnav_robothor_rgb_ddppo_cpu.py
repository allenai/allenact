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


class ObjectNavRoboThorRGBPPOExperimentConfig(ExperimentConfig):
    """An Object Navigation experiment configuration in RoboThor."""

    TRAIN_SCENES = [
        "FloorPlan_Train%d_%d" % (wall + 1, furniture + 1)
        for wall in range(1)
        for furniture in range(1)
    ]

    VALID_SCENES = TRAIN_SCENES
    TEST_SCENES = TRAIN_SCENES

    CAMERA_WIDTH = 400
    CAMERA_HEIGHT = 300

    SCREEN_SIZE = 224

    MAX_STEPS = 500

    # It also ignores the empirical success of all episodes with length > num_steps * ADVANCE_SCENE_ROLLOUT_PERIOD and
    # some with shorter lengths
    ADVANCE_SCENE_ROLLOUT_PERIOD = (
        10000000000000  # generally useful if more than 1 scene per worker
    )

    VALIDATION_SAMPLES_PER_SCENE = 10

    NUM_PROCESSES = 2

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
            "Remote",
            "SprayBottle",
            "Television",
            "Vase",
        ]
    )
    # TARGET_TYPES = sorted(["Television",])

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
                    "parallel": False,  # TODO False for debugging
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
        applyActionNoise=True,
        agentType="stochastic",
        rotateStepDegrees=45.0,
        visibilityDistance=1.5,
        gridSize=0.25,
        snapToGrid=False,
        agentMode="bot",
        include_private_scenes=False,
    )

    @classmethod
    def tag(cls):
        return "ObjectNavRobothorRGBPPO"

    @classmethod
    def training_pipeline(cls, **kwargs):
        ppo_steps = int(100)
        lr = 3e-4
        num_mini_batch = 1
        update_repeats = 3
        num_steps = 10
        save_interval = -1  # Never save
        log_interval = 1  # log every rollout
        gamma = 0.99
        use_gae = True
        gae_lambda = 0.95
        max_grad_norm = 0.5
        return TrainingPipeline(
            save_interval=save_interval,
            metric_accumulate_interval=log_interval,
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
                PipelineStage(loss_names=["ppo_loss"], max_stage_steps=ppo_steps),
                PipelineStage(loss_names=["ppo_loss"], max_stage_steps=ppo_steps),
            ],
            lr_scheduler_builder=Builder(
                LambdaLR, {"lr_lambda": LinearDecay(steps=2 * ppo_steps)}
            ),
        )

    def machine_params(self, mode="train", **kwargs):
        render_video = False

        sampler_devices = []
        if mode == "train":
            nprocesses = [1] * self.NUM_PROCESSES
            gpu_ids = ["cpu"] * len(nprocesses)
        elif mode == "valid":
            nprocesses = 0
            gpu_ids = []
        elif mode == "test":
            nprocesses = 2
            gpu_ids = [] if not torch.cuda.is_available() else [0]
            render_video = True
        else:
            raise NotImplementedError("mode must be 'train', 'valid', or 'test'.")

        # Disable parallelization for validation process
        if mode == "valid":
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
            "sampler_devices": sampler_devices
            if mode == "train"
            else gpu_ids,  # ignored with > 1 gpu_ids
            "observation_set": observation_set,
            "render_video": render_video,
        }

    @classmethod
    def create_model(cls, **kwargs) -> nn.Module:
        return ResnetTensorObjectNavActorCritic(
            action_space=gym.spaces.Discrete(len(ObjectNavTask.class_action_names())),
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
            "action_space": gym.spaces.Discrete(
                len(ObjectNavTask.class_action_names())
            ),
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
        res["scene_period"] = "manual"  # uses ADVANCE_SCENE_ROLLOUT_PERIOD
        res["env_args"] = {}
        res["env_args"].update(self.ENV_ARGS)
        try:
            res["env_args"]["x_display"] = (
                ("0.%d" % devices[process_ind % len(devices)])
                if devices is not None and len(devices) > 0
                else None
            )
        except Exception as _:
            pass
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
            self.VALID_SCENES,
            process_ind,
            total_processes,
            seeds=seeds,
            deterministic_cudnn=deterministic_cudnn,
        )
        res["scene_period"] = self.VALIDATION_SAMPLES_PER_SCENE
        res["max_tasks"] = self.VALIDATION_SAMPLES_PER_SCENE * len(res["scenes"])
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
        res["scene_period"] = self.VALIDATION_SAMPLES_PER_SCENE
        res["max_tasks"] = self.VALIDATION_SAMPLES_PER_SCENE * len(res["scenes"])
        res["env_args"] = {}
        res["env_args"].update(self.ENV_ARGS)
        try:
            res["env_args"]["x_display"] = (
                ("0.%d" % devices[process_ind % len(devices)])
                if devices is not None and len(devices) > 0
                else None
            )
        except Exception as _:
            pass
        return res
