from math import ceil
from typing import Dict, Any, List, Optional

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

from onpolicy_sync.losses.ppo import PPOConfig
from models.resnet_tensor_object_nav_models import ResnetTensorObjectNavActorCritic
from rl_ai2thor.ai2thor_sensors import RGBSensorThor, GoalObjectTypeThorSensor
from rl_robothor.object_nav.task_samplers import ObjectNavTaskSampler
from rl_robothor.object_nav.tasks import ObjectNavTask
from utils.experiment_utils import Builder, PipelineStage, TrainingPipeline
from onpolicy_sync.losses import PPO
from rl_base.experiment_config import ExperimentConfig
from rl_base.task import TaskSampler

from rl_base.preprocessor import ObservationSet
from rl_robothor.robothor_preprocessors import ResnetPreProcessorThor


class ObjectNavRoboThorExperimentConfig(ExperimentConfig):
    """An object navigation experiment in RoboTHOR."""

    OBJECT_TYPES = sorted(["Television", "Mug", "Apple", "AlarmClock", "BasketBall"])

    SCREEN_SIZE = 224

    SENSORS = [
        RGBSensorThor(
            {
                "height": SCREEN_SIZE,
                "width": SCREEN_SIZE,
                "use_resnet_normalization": True,
            }
        ),
        GoalObjectTypeThorSensor({"object_types": OBJECT_TYPES}),
    ]

    TRAIN_SCENES = [
        "FloorPlan_Train1_1"
        # "FloorPlan_Train%d_%d" % (wall, furniture)
        # for wall in range(1, 11)  # actual limit at 16
        # for furniture in range(1, 6)
    ]

    # VALID_SCENES = [
    #     "FloorPlan_RVal%d_%d" % (wall, furniture)
    #     for wall in range(1, 3)
    #     for furniture in range(1, 3)
    # ]
    VALID_SCENES = TRAIN_SCENES

    TEST_SCENES = VALID_SCENES

    VALIDATION_SAMPLES_PER_SCENE = 10

    TEST_SAMPLES_PER_SCENE = VALIDATION_SAMPLES_PER_SCENE

    PREPROCESSORS = [
        ResnetPreProcessorThor(
            config={
                "input_height": SCREEN_SIZE,
                "input_width": SCREEN_SIZE,
                "output_width": 7,
                "output_height": 7,
                "output_dims": 512,
                "torchvision_resnet_model": models.resnet18,
                "input_uuids": ["rgb"],
                "output_uuid": "rgb_resnet",
            }
        ),
    ]

    OBSERVATIONS = [
        "rgb_resnet",
        "goal_object_type_ind",
    ]

    ENV_ARGS = {
        "player_screen_height": SCREEN_SIZE,
        "player_screen_width": SCREEN_SIZE,
        "quality": "Very High",
        "rotation_step_degrees": 45,
    }

    MAX_STEPS = 100

    SCENE_PERIOD = 10

    @classmethod
    def tag(cls):
        return "ObjectNavRoboThor_1train_5tget"

    def training_pipeline(cls, **kwargs):
        ppo_steps = int(1e10)
        lr = 3e-5
        num_mini_batch = 3
        update_repeats = 5
        num_steps = 30
        log_interval = cls.MAX_STEPS * 50  # Log every 50 max length tasks
        save_interval = 100000  # Save every 100000 steps (approximately)
        gamma = 0.99
        use_gae = True
        gae_lambda = 1.0
        max_grad_norm = 0.3
        return TrainingPipeline(
            save_interval=save_interval,
            log_interval=log_interval,
            optimizer=Builder(optim.Adam, dict(lr=lr)),
            num_mini_batch=num_mini_batch,
            update_repeats=update_repeats,
            num_steps=num_steps,
            named_losses={"ppo_loss": Builder(PPO, dict(), default=PPOConfig,),},
            gamma=gamma,
            use_gae=use_gae,
            gae_lambda=gae_lambda,
            max_grad_norm=max_grad_norm,
            pipeline_stages=[
                PipelineStage(loss_names=["ppo_loss"], end_criterion=ppo_steps,),
            ],
        )

    def single_gpu(self):
        return 0

    def machine_params(self, mode="train", **kwargs):
        if mode == "train":
            nprocesses = 10
            gpu_ids = [] if not torch.cuda.is_available() else [self.single_gpu()]
        elif mode == "valid":
            nprocesses = 1
            gpu_ids = [] if not torch.cuda.is_available() else [self.single_gpu()]
        elif mode == "test":
            nprocesses = 1
            gpu_ids = [] if not torch.cuda.is_available() else [self.single_gpu()]
        else:
            raise NotImplementedError("mode must be 'train', 'valid', or 'test'.")

        observation_set = ObservationSet(
            self.OBSERVATIONS, self.PREPROCESSORS, self.SENSORS
        )

        return {
            "nprocesses": nprocesses,
            "gpu_ids": gpu_ids,
            "observation_set": observation_set,
        }

    def create_model(self, **kwargs) -> nn.Module:
        return ResnetTensorObjectNavActorCritic(
            action_space=gym.spaces.Discrete(len(ObjectNavTask.action_names())),
            observation_space=kwargs["observation_set"].observation_spaces,
            goal_sensor_uuid="goal_object_type_ind",
            rnn_hidden_size=512,
            goal_dims=32,
            compressor_hidden_out_dims=(128, 32),
            combiner_hidden_out_dims=(128, 32),
        )

    @staticmethod
    def make_sampler_fn(**kwargs) -> TaskSampler:
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
            "object_types": self.OBJECT_TYPES,
            "env_args": self.ENV_ARGS,
            "max_steps": self.MAX_STEPS,
            "sensors": self.SENSORS,
            "action_space": gym.spaces.Discrete(len(ObjectNavTask.action_names())),
            "seed": seeds[process_ind] if seeds is not None else None,
            "deterministic_cudnn": deterministic_cudnn,
            "rewards_config": {"step_penalty": -0.01, "goal_success_reward": 5},
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
        res["scene_period"] = self.SCENE_PERIOD
        res["env_args"]["x_display"] = "0.%d" % devices[0] if len(devices) > 0 else None
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
        res["scene_period"] = self.VALIDATION_SAMPLES_PER_SCENE
        res["max_tasks"] = self.VALIDATION_SAMPLES_PER_SCENE * len(res["scenes"])
        res["env_args"]["x_display"] = "0.%d" % devices[0] if len(devices) > 0 else None
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
        res["scene_period"] = self.TEST_SAMPLES_PER_SCENE
        res["max_tasks"] = self.TEST_SAMPLES_PER_SCENE * len(res["scenes"])
        res["env_args"]["x_display"] = "0.%d" % devices[0] if len(devices) > 0 else None
        return res
