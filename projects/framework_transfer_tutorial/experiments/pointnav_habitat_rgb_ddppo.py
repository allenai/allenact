from typing import Dict, Any, List, Optional

import gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
import habitat
from torchvision import models

from common.algorithms.onpolicy_sync.losses import PPO
from common.algorithms.onpolicy_sync.losses.ppo import PPOConfig
from projects.pointnav_baselines.models.point_nav_models import (
    ResnetTensorPointNavActorCritic,
)
from plugins.rl_habitat.habitat_sensors import (
    RGBSensorHabitat,
    TargetCoordinatesSensorHabitat,
)
from rl_base.experiment_config import ExperimentConfig
from rl_base.preprocessor import ObservationSet
from rl_base.task import TaskSampler
from plugins.rl_habitat.habitat_preprocessors import ResnetPreProcessorHabitat
from plugins.rl_habitat.habitat_task_samplers import PointNavTaskSampler
from plugins.rl_habitat.habitat_utils import construct_env_configs
from plugins.rl_robothor.robothor_tasks import PointNavTask
from utils.experiment_utils import Builder, PipelineStage, TrainingPipeline, LinearDecay


class ObjectNavRoboThorRGBPPOExperimentConfig(ExperimentConfig):
    """A Point Navigation experiment configuration in RoboThor."""

    # Task Parameters
    MAX_STEPS = 500
    REWARD_CONFIG = {
        "step_penalty": -0.01,
        "goal_success_reward": 10.0,
        "failed_stop_reward": 0.0,
        "shaping_weight": 1.0,
    }

    # Simulator Parameters
    CAMERA_WIDTH = 640
    CAMERA_HEIGHT = 480
    SCREEN_SIZE = 224

    # Training Engine Parameters
    ADVANCE_SCENE_ROLLOUT_PERIOD = 10000000000000
    NUM_PROCESSES = 60
    TRAINING_GPUS = [0, 1, 2, 3, 4, 5, 6]
    VALIDATION_GPUS = [7]
    TESTING_GPUS = [7]

    TRAIN_SCENES = (
        "habitat/habitat-api/data/datasets/pointnav/gibson/v1/train/train.json.gz"
    )
    VALID_SCENES = (
        "habitat/habitat-api/data/datasets/pointnav/gibson/v1/val/val.json.gz"
    )
    TEST_SCENES = (
        "habitat/habitat-api/data/datasets/pointnav/gibson/v1/test/test.json.gz"
    )

    TRAIN_GPUS = [0, 1, 2, 3, 4, 5, 6, 7]
    VALIDATION_GPUS = [7]
    TESTING_GPUS = [7]

    NUM_PROCESSES = 80
    CONFIG = habitat.get_config("configs/gibson.yaml")
    CONFIG.defrost()
    CONFIG.NUM_PROCESSES = NUM_PROCESSES
    CONFIG.SIMULATOR_GPU_IDS = TRAIN_GPUS
    CONFIG.DATASET.SCENES_DIR = "habitat/habitat-api/data/scene_datasets/"
    CONFIG.DATASET.POINTNAVV1.CONTENT_SCENES = ["*"]
    CONFIG.DATASET.DATA_PATH = TRAIN_SCENES
    CONFIG.SIMULATOR.AGENT_0.SENSORS = ["RGB_SENSOR"]
    CONFIG.SIMULATOR.RGB_SENSOR.WIDTH = CAMERA_WIDTH
    CONFIG.SIMULATOR.RGB_SENSOR.HEIGHT = CAMERA_HEIGHT
    CONFIG.SIMULATOR.TURN_ANGLE = 30
    CONFIG.SIMULATOR.FORWARD_STEP_SIZE = 0.25
    CONFIG.ENVIRONMENT.MAX_EPISODE_STEPS = MAX_STEPS

    CONFIG.TASK.TYPE = "Nav-v0"
    CONFIG.TASK.SUCCESS_DISTANCE = 0.2
    CONFIG.TASK.SENSORS = ["POINTGOAL_WITH_GPS_COMPASS_SENSOR"]
    CONFIG.TASK.POINTGOAL_WITH_GPS_COMPASS_SENSOR.GOAL_FORMAT = "POLAR"
    CONFIG.TASK.POINTGOAL_WITH_GPS_COMPASS_SENSOR.DIMENSIONALITY = 2
    CONFIG.TASK.GOAL_SENSOR_UUID = "pointgoal_with_gps_compass"
    CONFIG.TASK.MEASUREMENTS = ["DISTANCE_TO_GOAL", "SPL"]
    CONFIG.TASK.SPL.TYPE = "SPL"
    CONFIG.TASK.SPL.SUCCESS_DISTANCE = 0.2

    CONFIG.MODE = "train"

    SENSORS = [
        RGBSensorHabitat(
            height=SCREEN_SIZE, width=SCREEN_SIZE, use_resnet_normalization=True,
        ),
        TargetCoordinatesSensorHabitat(coordinate_dims=2),
    ]

    PREPROCESSORS = [
        Builder(
            ResnetPreProcessorHabitat,
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
                "parallel": False,  # TODO False for debugging
            },
        ),
    ]

    OBSERVATIONS = [
        "rgb_resnet",
        "target_coordinates_ind",
    ]

    TRAIN_CONFIGS = construct_env_configs(CONFIG)

    @classmethod
    def tag(cls):
        return "PointNavHabitatRGBPPO"

    @classmethod
    def training_pipeline(cls, **kwargs):
        ppo_steps = int(250000000)
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
                PipelineStage(loss_names=["ppo_loss"], max_stage_steps=ppo_steps)
            ],
            lr_scheduler_builder=Builder(
                LambdaLR, {"lr_lambda": LinearDecay(steps=ppo_steps)}
            ),
        )

    def split_num_processes(self, ndevices):
        assert self.NUM_PROCESSES >= ndevices, "NUM_PROCESSES {} < ndevices".format(
            self.NUM_PROCESSES, ndevices
        )
        res = [0] * ndevices
        for it in range(self.NUM_PROCESSES):
            res[it % ndevices] += 1
        return res

    def machine_params(self, mode="train", **kwargs):
        if mode == "train":
            workers_per_device = 1
            gpu_ids = (
                []
                if not torch.cuda.is_available()
                else self.TRAINING_GPUS * workers_per_device
            )
            nprocesses = (
                1
                if not torch.cuda.is_available()
                else self.split_num_processes(len(gpu_ids))
            )
            sampler_devices = self.TRAINING_GPUS
            render_video = False
        elif mode == "valid":
            nprocesses = 1
            gpu_ids = [] if not torch.cuda.is_available() else self.VALIDATION_GPUS
            render_video = False
        elif mode == "test":
            nprocesses = 1
            gpu_ids = [] if not torch.cuda.is_available() else self.TESTING_GPUS
            render_video = False
        else:
            raise NotImplementedError("mode must be 'train', 'valid', or 'test'.")

        # Disable parallelization for validation process
        if mode == "valid":
            for prep in self.PREPROCESSORS:
                prep.kwargs["parallel"] = False

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
            "render_video": render_video,
        }

    # Define Model
    @classmethod
    def create_model(cls, **kwargs) -> nn.Module:
        return ResnetTensorPointNavActorCritic(
            action_space=gym.spaces.Discrete(len(PointNavTask.class_action_names())),
            observation_space=kwargs["observation_set"].observation_spaces,
            goal_sensor_uuid="target_coordinates_ind",
            rgb_resnet_preprocessor_uuid="rgb_resnet",
            hidden_size=512,
            goal_dims=32,
        )

    # Define Task Sampler
    @classmethod
    def make_sampler_fn(cls, **kwargs) -> TaskSampler:
        return PointNavTaskSampler(**kwargs)

    def train_task_sampler_args(
        self,
        process_ind: int,
        total_processes: int,
        devices: Optional[List[int]] = None,
        seeds: Optional[List[int]] = None,
        deterministic_cudnn: bool = False,
    ) -> Dict[str, Any]:
        config = self.TRAIN_CONFIGS[process_ind]
        return {
            "env_config": config,
            "max_steps": self.MAX_STEPS,
            "sensors": self.SENSORS,
            "action_space": gym.spaces.Discrete(len(PointNavTask.action_names())),
            "distance_to_goal": self.DISTANCE_TO_GOAL,
        }

    def valid_task_sampler_args(
        self,
        process_ind: int,
        total_processes: int,
        devices: Optional[List[int]] = None,
        seeds: Optional[List[int]] = None,
        deterministic_cudnn: bool = False,
    ) -> Dict[str, Any]:
        config = self.CONFIG.clone()
        config.defrost()
        config.DATASET.DATA_PATH = self.VALID_SCENES
        config.MODE = "validate"
        config.freeze()
        return {
            "env_config": config,
            "max_steps": self.MAX_STEPS,
            "sensors": self.SENSORS,
            "action_space": gym.spaces.Discrete(len(PointNavTask.action_names())),
            "distance_to_goal": self.DISTANCE_TO_GOAL,
        }

    def test_task_sampler_args(
        self,
        process_ind: int,
        total_processes: int,
        devices: Optional[List[int]] = None,
        seeds: Optional[List[int]] = None,
        deterministic_cudnn: bool = False,
    ) -> Dict[str, Any]:
        config = self.TEST_CONFIGS[process_ind]
        return {
            "env_config": config,
            "max_steps": self.MAX_STEPS,
            "sensors": self.SENSORS,
            "action_space": gym.spaces.Discrete(len(PointNavTask.action_names())),
            "distance_to_goal": self.DISTANCE_TO_GOAL,
        }
