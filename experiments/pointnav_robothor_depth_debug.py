from typing import Dict, Any, List, Optional
from math import ceil, gcd

import gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
import numpy as np
from torchvision import models as models

from onpolicy_sync.losses.ppo import PPOConfig
from models.point_nav_models import PointNavActorCriticTrainResNet50GRU
from onpolicy_sync.losses import PPO
from rl_base.experiment_config import ExperimentConfig
from rl_base.task import TaskSampler
from rl_base.preprocessor import ObservationSet
from rl_robothor.robothor_tasks import PointNavTask
from rl_robothor.robothor_task_samplers import PointNavTaskSampler
from rl_robothor.robothor_sensors import GPSCompassSensorRoboThor, DepthSensorRoboThor
from rl_habitat.habitat_preprocessors import ResnetPreProcessorHabitat
from utils.experiment_utils import Builder, PipelineStage, TrainingPipeline, LinearDecay


class PointNavRoboThorDepthPPOExperimentConfig(ExperimentConfig):
    """A Point Navigation experiment configuration in RoboThor"""

    TRAIN_SCENES = [
        "FloorPlan_Train2_3", "FloorPlan_Train5_1"
    ]

    # VALID_SCENES = [
    #     "FloorPlan_Val%d_%d" % (wall + 1, furniture + 1)
    #     for wall in range(3)
    #     for furniture in range(5)
    # ]
    #

    SCREEN_SIZE = 256
    MAX_STEPS = 500
    ADVANCE_SCENE_ROLLOUT_PERIOD = 6  # if more than 1 scene per sampler

    VALIDATION_SAMPLES_PER_SCENE = 1

    NUM_PROCESSES = 2

    SENSORS = [
        DepthSensorRoboThor(
            {
                "height": SCREEN_SIZE,
                "width": SCREEN_SIZE,
                "use_resnet_normalization": True,
            }
        ),
        GPSCompassSensorRoboThor({}),
    ]

    PREPROCESSORS = [
        Builder(ResnetPreProcessorHabitat,
                dict(config={
                    "input_height": SCREEN_SIZE,
                    "input_width": SCREEN_SIZE,
                    "output_width": 8,
                    "output_height": 8,
                    "output_dims": 2048,
                    "pool": False,
                    "torchvision_resnet_model": models.resnet50,
                    "input_uuids": ["depth"],
                    "output_uuid": "depth_resnet",
                    "parallel": False,  # TODO False for debugging
            })
        ),
    ]

    OBSERVATIONS = [
        "depth_resnet",
        "target_coordinates_ind",
    ]

    ENV_ARGS = dict(
        width=max(300, SCREEN_SIZE),
        height=max(300, SCREEN_SIZE),
        fieldOfView=79.0,
        continuousMode=True,
        applyActionNoise=True,
        agentType="stochastic",  # to avoid getting locked with 45 degrees (in old ai2thor)
        rotateStepDegrees=45.0,
        visibilityDistance=1.0,  # 1.0 vs 1.5 in CVPR robothor
        gridSize=0.25,
        snapToGrid=False,
        agentMode="bot",
        movementGaussianMu=1e-20,  # almost deterministic
        movementGaussianSigma=1e-20,  # almost deterministic
        rotateGaussianMu=1e-20,  # almost deterministic
        rotateGaussianSigma=1e-20,  # almost deterministic
        renderDepthImage=True,
    )

    @classmethod
    def tag(cls):
        return "PointNavRoboThorDepthPPO"

    @classmethod
    def training_pipeline(cls, **kwargs):
        ppo_steps = int(7.5e7)
        lr = 1e-4  # 2.5e-4 for 12 workers, 1 batch
        num_mini_batch = 1  # 31 fps for 3 (36 workers), 6 (6GB with 36 workers)
        update_repeats = 1
        num_steps = 128
        save_interval = 50000
        log_interval = 1  # log every rollout
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
            named_losses={"ppo_loss": Builder(PPO, kwargs={"use_clipped_value_loss": True}, default=PPOConfig,)},
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

    def machine_params(self, mode="train", **kwargs):
        if mode == "train":
            gpu_ids = [] if not torch.cuda.is_available() else [0]
            nprocesses = 1 if not torch.cuda.is_available() else self.NUM_PROCESSES
            sampler_devices = [0]
            render_video = False
        elif mode == "valid":
            nprocesses = 0
            if not torch.cuda.is_available():
                gpu_ids = []
            else:
                gpu_ids = [1]
            render_video = False
        elif mode == "test":
            nprocesses = 1
            if not torch.cuda.is_available():
                gpu_ids = []
            else:
                gpu_ids = [0]
            render_video = True
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
            "sampler_devices": sampler_devices if mode == "train" else gpu_ids,
            "observation_set": observation_set,
            "render_video": render_video,
        }

    @classmethod
    def create_model(cls, **kwargs) -> nn.Module:
        return PointNavActorCriticTrainResNet50GRU(
            action_space=gym.spaces.Discrete(len(PointNavTask.action_names())),
            observation_space=kwargs["observation_set"].observation_spaces,
            goal_sensor_uuid="target_coordinates_ind",
            hidden_size=512,
            embed_coordinates=False,
            coordinate_embedding_dim=8,
            coordinate_dims=2,
            num_rnn_layers=1,
            rnn_type='GRU'
        )

    @classmethod
    def make_sampler_fn(cls, **kwargs) -> TaskSampler:
        return PointNavTaskSampler(**kwargs)

    @staticmethod
    def _partition_inds(n: int, num_parts: int):
        return np.round(np.linspace(0, n, num_parts + 1, endpoint=True)).astype(
            np.int32
        )

    @staticmethod
    def lcm(a, b):
        return abs(a * b) // gcd(a, b)

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
            "sensors": self.SENSORS,
            "max_steps": self.MAX_STEPS,
            "action_space": gym.spaces.Discrete(len(PointNavTask.action_names())),
            "seed": seeds[process_ind] if seeds is not None else None,
            "deterministic_cudnn": deterministic_cudnn,
            "rewards_config": {
                "step_penalty": -0.01,
                "goal_success_reward": 10.0,
                "unsuccessful_action_penalty": 0.0,
                "failed_stop_reward": 0.0,
                "shaping_weight": 1.0,  # applied to the decrease in distance to target
                "exploration_shaping_weight": 0.0,  # relative to shaping weight
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
        scenes = self.TRAIN_SCENES * (self.lcm(len(self.TRAIN_SCENES), self.NUM_PROCESSES) // len(self.TRAIN_SCENES))
        scenes_per_sampler = len(scenes) / self.NUM_PROCESSES
        res = self._get_sampler_args_for_scene_split(
            scenes,
            process_ind,
            total_processes,
            seeds=seeds,
            deterministic_cudnn=deterministic_cudnn,
        )

        assert len(res["scenes"]) <= len(self.TRAIN_SCENES), "assigned {} to sampler out of {} scenes".format(
            len(res["scenes"]), len([self.TRAIN_SCENES])
        )
        assert len(res["scenes"]) == scenes_per_sampler, "assigned {} to sampler for expected {} scenes".format(
            len(res["scenes"]), scenes_per_sampler
        )

        res["scene_period"] = "manual"
        res["env_args"] = {}
        res["env_args"].update(self.ENV_ARGS)
        res["env_args"]["x_display"] = (
            ("0.%d" % devices[process_ind % len(devices)]) if devices is not None and len(devices) > 0 else None
        )
        res["allow_flipping"] = False
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
            ("0.%d" % devices[process_ind % len(devices)]) if devices is not None and len(devices) > 0 else None
        )
        return res
