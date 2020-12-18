from math import ceil
from typing import Dict, Any, List, Optional, Sequence

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

from core.algorithms.onpolicy_sync.losses import PPO
from core.algorithms.onpolicy_sync.losses.ppo import PPOConfig
from core.base_abstractions.experiment_config import ExperimentConfig
from core.base_abstractions.sensor import SensorSuite
from core.base_abstractions.task import TaskSampler
from plugins.robothor_plugin.robothor_sensors import RGBSensorMultiRoboThor
from plugins.robothor_plugin.robothor_task_samplers import NavToPartnerTaskSampler
from plugins.robothor_plugin.robothor_tasks import NavToPartnerTask
from plugins.robothor_plugin.robothor_models import NavToPartnerActorCriticSimpleConvRNN
from utils.experiment_utils import Builder, PipelineStage, TrainingPipeline, LinearDecay
from utils.viz_utils import VizSuite, AgentViewViz
from utils.multi_agent_viz_utils import MultiTrajectoryViz
from plugins.robothor_plugin.robothor_viz import ThorMultiViz


class NavToPartnerRoboThorRGBPPOExperimentConfig(ExperimentConfig):
    """A Multi-Agent Navigation experiment configuration in RoboThor."""

    # Task Parameters
    MAX_STEPS = 500
    REWARD_CONFIG = {
        "step_penalty": -0.01,
        "max_success_distance": 0.75,
        "success_reward": 5.0,
    }

    # Simulator Parameters
    CAMERA_WIDTH = 300
    CAMERA_HEIGHT = 300
    SCREEN_SIZE = 224

    # Training Engine Parameters
    ADVANCE_SCENE_ROLLOUT_PERIOD: Optional[int] = None
    NUM_PROCESSES = 20
    TRAINING_GPUS: List[int] = [0]
    VALIDATION_GPUS: List[int] = [0]
    TESTING_GPUS: List[int] = [0]

    SENSORS = [
        RGBSensorMultiRoboThor(
            agent_count=2,
            height=SCREEN_SIZE,
            width=SCREEN_SIZE,
            use_resnet_normalization=True,
            uuid="rgb",
        ),
    ]

    OBSERVATIONS = [
        "rgb",
    ]

    ENV_ARGS = dict(
        width=CAMERA_WIDTH,
        height=CAMERA_HEIGHT,
        rotateStepDegrees=30.0,
        visibilityDistance=1.0,
        gridSize=0.25,
        agentCount=2,
    )

    @classmethod
    def tag(cls):
        return "NavToPartnerRobothorRGBPPO"

    @classmethod
    def training_pipeline(cls, **kwargs):
        ppo_steps = int(1000000)
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
            metric_accumulate_interval=log_interval,
            optimizer_builder=Builder(optim.Adam, dict(lr=lr)),
            num_mini_batch=num_mini_batch,
            update_repeats=update_repeats,
            max_grad_norm=max_grad_norm,
            num_steps=num_steps,
            named_losses={"ppo_loss": PPO(**PPOConfig)},
            gamma=gamma,
            use_gae=use_gae,
            gae_lambda=gae_lambda,
            advance_scene_rollout_period=cls.ADVANCE_SCENE_ROLLOUT_PERIOD,
            pipeline_stages=[
                PipelineStage(loss_names=["ppo_loss"], max_stage_steps=ppo_steps)
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

    viz: Optional[VizSuite] = None

    def get_viz(self, mode):
        if self.viz is not None:
            return self.viz

        self.viz = VizSuite(
            mode=mode,
            # Basic 2D trajectory visualizer (task output source):
            base_trajectory=MultiTrajectoryViz(),  # plt_colormaps=["cool", "cool"]),
            # Egocentric view visualizer (vector task source):
            egeocentric=AgentViewViz(max_video_length=100, max_episodes_in_group=1),
            # Specialized 2D trajectory visualizer (task output source):
            thor_trajectory=ThorMultiViz(
                figsize=(16, 8),
                viz_rows_cols=(448, 448),
                scenes=("FloorPlan_Train{}_{}", 1, 1, 1, 1),
            ),
        )

        return self.viz

    def machine_params(self, mode="train", **kwargs):
        visualizer = None
        if mode == "train":
            devices = (
                ["cpu"] if not torch.cuda.is_available() else list(self.TRAINING_GPUS)
            )
            nprocesses = (
                4
                if not torch.cuda.is_available()
                else self.split_num_processes(len(devices))
            )
        elif mode == "valid":
            nprocesses = 0
            devices = ["cpu"] if not torch.cuda.is_available() else self.VALIDATION_GPUS
        elif mode == "test":
            nprocesses = 1
            devices = ["cpu"] if not torch.cuda.is_available() else self.TESTING_GPUS
            visualizer = self.get_viz(mode=mode)
        else:
            raise NotImplementedError("mode must be 'train', 'valid', or 'test'.")

        return {
            "nprocesses": nprocesses,
            "devices": devices,
            "visualizer": visualizer,
        }

    # TODO Define Model
    @classmethod
    def create_model(cls, **kwargs) -> nn.Module:
        return NavToPartnerActorCriticSimpleConvRNN(
            action_space=gym.spaces.Discrete(
                len(NavToPartnerTask.class_action_names())
            ),
            observation_space=SensorSuite(cls.SENSORS).observation_spaces,
            hidden_size=512,
        )

    # Define Task Sampler
    @classmethod
    def make_sampler_fn(cls, **kwargs) -> TaskSampler:
        return NavToPartnerTaskSampler(**kwargs)

    # Utility Functions for distributing scenes between GPUs
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
            "max_steps": self.MAX_STEPS,
            "sensors": self.SENSORS,
            "action_space": gym.spaces.Discrete(
                len(NavToPartnerTask.class_action_names())
            ),
            "seed": seeds[process_ind] if seeds is not None else None,
            "deterministic_cudnn": deterministic_cudnn,
            "rewards_config": self.REWARD_CONFIG,
        }

    def train_task_sampler_args(
        self,
        process_ind: int,
        total_processes: int,
        devices: Optional[List[int]] = None,
        seeds: Optional[List[int]] = None,
        deterministic_cudnn: bool = False,
    ) -> Dict[str, Any]:
        scenes = ["FloorPlan_Train1_1"]

        res = self._get_sampler_args_for_scene_split(
            scenes,
            process_ind,
            total_processes,
            seeds=seeds,
            deterministic_cudnn=deterministic_cudnn,
        )
        res["env_args"] = {
            **self.ENV_ARGS,
            "x_display": ("0.%d" % devices[process_ind % len(devices)])
            if devices is not None and len(devices) > 0
            else None,
        }
        return res

    def valid_task_sampler_args(
        self,
        process_ind: int,
        total_processes: int,
        devices: Optional[List[int]] = None,
        seeds: Optional[List[int]] = None,
        deterministic_cudnn: bool = False,
    ) -> Dict[str, Any]:
        scenes = ["FloorPlan_Train1_1"]

        res = self._get_sampler_args_for_scene_split(
            scenes,
            process_ind,
            total_processes,
            seeds=seeds,
            deterministic_cudnn=deterministic_cudnn,
        )
        res["env_args"] = {
            **self.ENV_ARGS,
            "x_display": ("0.%d" % devices[process_ind % len(devices)])
            if devices is not None and len(devices) > 0
            else None,
        }
        res["max_tasks"] = 20
        return res

    def test_task_sampler_args(
        self,
        process_ind: int,
        total_processes: int,
        devices: Optional[List[int]] = None,
        seeds: Optional[List[int]] = None,
        deterministic_cudnn: bool = False,
    ) -> Dict[str, Any]:
        scenes = ["FloorPlan_Train1_1"]

        res = self._get_sampler_args_for_scene_split(
            scenes,
            process_ind,
            total_processes,
            seeds=seeds,
            deterministic_cudnn=deterministic_cudnn,
        )
        res["env_args"] = {
            **self.ENV_ARGS,
            "x_display": ("0.%d" % devices[process_ind % len(devices)])
            if devices is not None and len(devices) > 0
            else None,
        }
        res["max_tasks"] = 4
        return res
