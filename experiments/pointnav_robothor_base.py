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

from onpolicy_sync.losses.ppo import PPOConfig
from models.point_nav_models import PointNavActorCriticResNet50GRU
from onpolicy_sync.losses import PPO
from rl_base.experiment_config import ExperimentConfig
from rl_base.sensor import SensorSuite
from rl_base.task import TaskSampler
from rl_base.preprocessor import ObservationSet
from rl_robothor.robothor_tasks import PointNavTask
from rl_robothor.robothor_task_samplers import PointNavTaskSampler
from rl_robothor.robothor_sensors import GPSCompassSensorRoboThor
from rl_ai2thor.ai2thor_sensors import RGBSensorThor
from rl_habitat.habitat_preprocessors import ResnetPreProcessorHabitat
from utils.experiment_utils import Builder, PipelineStage, TrainingPipeline, LinearDecay


class PointNavRoboThorBaseExperimentConfig(ExperimentConfig):
    """A Point Navigation experiment configuraqtion in RoboThor"""

    TRAIN_SCENES = [
        "FloorPlan_Train%d_%d" % (wall, furniture)
        for wall in range(1, 13)
        for furniture in range(1, 6)
    ]

    VALID_SCENES = [
        "FloorPlan_Val%d_%d" % (wall, furniture)
        for wall in range(1, 4)
        for furniture in range(1, 6)
    ]

    VALIDATION_SAMPLES_PER_SCENE = 1

    SCREEN_SIZE = 256
    MAX_STEPS = 500
    ADVANCE_SCENE_ROLLOUT_PERIOD = 10  # if more than 1 scene per worker

    NUM_PROCESSES = 10

    SENSORS = [
        RGBSensorThor(
            {
                "height": SCREEN_SIZE,
                "width": SCREEN_SIZE,
                "use_resnet_normalization": True,
            }
        ),
        GPSCompassSensorRoboThor({}),
    ]

    PREPROCESSORS = [
        ResnetPreProcessorHabitat(
            config={
                "input_height": SCREEN_SIZE,
                "input_width": SCREEN_SIZE,
                "output_width": 1,
                "output_height": 1,
                "output_dims": 2048,
                "pool": True,
                "torchvision_resnet_model": models.resnet50,
                "input_uuids": ["rgb"],
                "output_uuid": "rgb_resnet",
            }
        ),
    ]

    OBSERVATIONS = [
        "rgb_resnet",
        "target_coordinates_ind",
    ]

    ENV_ARGS = dict(
        width=max(300, SCREEN_SIZE),
        height=max(300, SCREEN_SIZE),
        fieldOfView=79.0,
        continuousMode=True,
        applyActionNoise=True,
        agentType="stochastic",
        rotateStepDegrees=45.0,
        visibilityDistance=1.0,
        gridSize=0.25,
        snapToGrid=False,
        agentMode="bot",
        movementGaussianMu=1e-20,
        movementGaussianSigma=1e-20,
        rotateGaussianMu=1e-20,
        rotateGaussianSigma=1e-20,
    )

    @classmethod
    def tag(cls):
        return "PointNav"

    @classmethod
    def training_pipeline(cls, **kwargs):
        ppo_steps = 7.5e7
        lr = 2.5e-4
        num_mini_batch = 1
        update_repeats = 4
        num_steps = 64
        save_interval = 1000000
        log_interval = 2000
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
            # nprocesses = 1 if not torch.cuda.is_available() else self.NUM_PROCESSES
            # sampler_devices = [0, 1, 2, 3, 4, 5, 6, 7]
            # gpu_ids = [] if not torch.cuda.is_available() else [7]
            # render_video = False
            nprocesses = 1 if not torch.cuda.is_available() else self.NUM_PROCESSES
            sampler_devices = [0]
            gpu_ids = [] if not torch.cuda.is_available() else [0]
            render_video = False
        elif mode == "valid":
            nprocesses = 1
            if not torch.cuda.is_available():
                gpu_ids = []
            else:
                gpu_ids = [0]
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

        observation_set = ObservationSet(
            self.OBSERVATIONS, self.PREPROCESSORS, self.SENSORS
        )

        return {
            "nprocesses": nprocesses,
            "gpu_ids": gpu_ids,
            "sampler_devices": sampler_devices if mode == "train" else gpu_ids,
            "observation_set": observation_set,
            "render_video": render_video,
        }

    @classmethod
    def create_model(cls, **kwargs) -> nn.Module:
        return PointNavActorCriticResNet50GRU(
            action_space=gym.spaces.Discrete(len(PointNavTask.action_names())),
            observation_space=SensorSuite(cls.SENSORS).observation_spaces,
            goal_sensor_uuid="target_coordinates_ind",
            hidden_size=512,
            embed_coordinates=False,
            coordinate_dims=2,
        )

    @classmethod
    def make_sampler_fn(cls, **kwargs) -> TaskSampler:
        return PointNavTaskSampler(**kwargs)

    @staticmethod
    def _partition_inds(n: int, num_parts: int):
        return np.round(np.linspace(0, n, num_parts + 1, endpoint=True)).astype(
            np.int32
        )

    def _get_sampler_args_for_scene_split(
        self,
        scenes: List[str],
        # scene_to_episodes: Dict[str, Any],
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
            # "scenes": {scene: scene_to_episodes[scene] for scene in scenes[inds[process_ind] : inds[process_ind + 1]]},
            "scenes": scenes[inds[process_ind]: inds[process_ind + 1]],
            "max_steps": self.MAX_STEPS,
            "sensors": self.SENSORS,
            "action_space": gym.spaces.Discrete(len(PointNavTask.action_names())),
            "seed": seeds[process_ind] if seeds is not None else None,
            "deterministic_cudnn": deterministic_cudnn,
            "rewards_config": {
                "step_penalty": -0.01,
                "goal_success_reward": 5.0,
                "unsuccessful_action_penalty": -0.05,
                "failed_stop_reward": -1.0,
                "shaping_weight": 0.25,  # applied to the decrease in distance to target
                "exploration_shaping_weight": 0.1,  # relative to shaping weight
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
        # if self.TRAIN_SCENES is None:
        #     with open('data/dataset_pointnav.json', 'r') as f:
        #         all_scenes = json.load(f)
        #         self.TRAIN_SCENES = {}
        #         for episode in all_scenes:
        #             if episode['scene'] in self.TRAIN_SCENES:
        #                 self.TRAIN_SCENES[episode['scene']].append(episode)
        #             else:
        #                 self.TRAIN_SCENES[episode['scene']] = [episode]
        #         self.scene_names = sorted(self.TRAIN_SCENES.keys())
        #         print("Loaded episodes for scenes {}".format(self.scene_names))
        #
        res = self._get_sampler_args_for_scene_split(
            # self.scene_names,
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
            ("0.%d" % devices[process_ind % len(devices)]) if devices is not None and len(devices) > 0 else None
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
        res["scene_period"] = self.VALIDATION_SAMPLES_PER_SCENE
        res["max_tasks"] = self.VALIDATION_SAMPLES_PER_SCENE * len(res["scenes"])
        res["env_args"] = {}
        res["env_args"].update(self.ENV_ARGS)
        res["env_args"]["x_display"] = (
            ("0.%d" % devices[process_ind % len(devices)]) if devices is not None and len(devices) > 0 else None
        )
        return res
