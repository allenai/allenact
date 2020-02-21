from typing import Dict, Any, List, Optional

import gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torchvision import models

import habitat
from onpolicy_sync.losses.ppo import PPOConfig
from models.point_nav_models import PointNavActorCriticResNet50GRU
from onpolicy_sync.losses import PPO
from rl_base.experiment_config import ExperimentConfig
from rl_base.sensor import SensorSuite
from rl_base.task import TaskSampler
from rl_base.preprocessor import ObservationSet
from rl_habitat.habitat_tasks import PointNavTask
from rl_habitat.habitat_task_samplers import PointNavTaskSampler
from rl_habitat.habitat_sensors import RGBSensorHabitat, TargetCoordinatesSensorHabitat
from rl_habitat.habitat_preprocessors import ResnetPreProcessorHabitat
from rl_habitat.habitat_utils import construct_env_configs
from utils.experiment_utils import Builder, PipelineStage, TrainingPipeline, LinearDecay


class PointNavHabitatRGBDeterministicResNet50GRUPPOExperimentConfig(ExperimentConfig):
    """A Point Navigation experiment configuraqtion in Habitat"""

    TRAIN_SCENES = "habitat/habitat-api/data/datasets/pointnav/gibson/v1/train/train.json.gz"
    VALID_SCENES = "habitat/habitat-api/data/datasets/pointnav/gibson/v1/val/val.json.gz"
    TEST_SCENES = "habitat/habitat-api/data/datasets/pointnav/gibson/v1/test/test.json.gz"

    SCREEN_SIZE = 256
    MAX_STEPS = 500
    DISTANCE_TO_GOAL = 0.2

    NUM_PROCESSES = 32

    SENSORS = [
        RGBSensorHabitat(
            {
                "height": SCREEN_SIZE,
                "width": SCREEN_SIZE,
                "use_resnet_normalization": True,
            }
        ),
        TargetCoordinatesSensorHabitat({"coordinate_dims": 2}),
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

    CONFIG = habitat.get_config('configs/gibson.yaml')
    CONFIG.defrost()
    CONFIG.NUM_PROCESSES = NUM_PROCESSES
    # CONFIG.SIMULATOR_GPU_ID = 0
    CONFIG.SIMULATOR_GPU_IDS = [1,2,3,4,5,6,7]
    CONFIG.DATASET.SCENES_DIR = 'habitat/habitat-api/data/scene_datasets/'
    CONFIG.DATASET.POINTNAVV1.CONTENT_SCENES = ['*']
    CONFIG.DATASET.DATA_PATH = TRAIN_SCENES
    CONFIG.SIMULATOR.AGENT_0.SENSORS = ['RGB_SENSOR']
    CONFIG.SIMULATOR.RGB_SENSOR.WIDTH = SCREEN_SIZE
    CONFIG.SIMULATOR.RGB_SENSOR.HEIGHT = SCREEN_SIZE
    CONFIG.SIMULATOR.TURN_ANGLE = 45
    CONFIG.SIMULATOR.FORWARD_STEP_SIZE = 0.25
    CONFIG.ENVIRONMENT.MAX_EPISODE_STEPS = MAX_STEPS

    CONFIG.TASK.TYPE = 'Nav-v0'
    CONFIG.TASK.SUCCESS_DISTANCE = DISTANCE_TO_GOAL
    CONFIG.TASK.SENSORS = ['POINTGOAL_WITH_GPS_COMPASS_SENSOR']
    CONFIG.TASK.POINTGOAL_WITH_GPS_COMPASS_SENSOR.GOAL_FORMAT = "POLAR"
    CONFIG.TASK.POINTGOAL_WITH_GPS_COMPASS_SENSOR.DIMENSIONALITY = 2
    CONFIG.TASK.GOAL_SENSOR_UUID = 'pointgoal_with_gps_compass'
    CONFIG.TASK.MEASUREMENTS = ['DISTANCE_TO_GOAL', 'SPL']
    CONFIG.TASK.SPL.TYPE = 'SPL'
    CONFIG.TASK.SPL.SUCCESS_DISTANCE = 0.2

    # TRAIN_CONFIGS = construct_env_configs(CONFIG)

    @classmethod
    def tag(cls):
        return "PointNav"

    @classmethod
    def training_pipeline(cls, **kwargs):
        ppo_steps = 7.5e7
        lr = 2.5e-4
        num_mini_batch = 1
        update_repeats = 4
        num_steps = 128
        save_interval = 1000000
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
            named_losses={"ppo_loss": Builder(PPO, kwargs={"use_clipped_value_loss": True}, default=PPOConfig,)},
            gamma=gamma,
            use_gae=use_gae,
            gae_lambda=gae_lambda,
            advance_scene_rollout_period=None,
            pipeline_stages=[
                PipelineStage(loss_names=["ppo_loss"], end_criterion=ppo_steps)
            ],
            lr_scheduler_builder=Builder(
                LambdaLR, {"lr_lambda": LinearDecay(steps=ppo_steps)}
            ),
        )

    @classmethod
    def evaluation_params(cls, **kwargs):
        nprocesses = 1
        gpu_ids = [] if not torch.cuda.is_available() else [1]
        res = cls.training_pipeline()
        del res["pipeline"]
        del res["optimizer"]
        res["nprocesses"] = nprocesses
        res["gpu_ids"] = gpu_ids
        return res

    def machine_params(self, mode="train", **kwargs):
        if mode == "train":
            nprocesses = 1 if not torch.cuda.is_available() else self.NUM_PROCESSES
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

    # def _get_sampler_args(
    #     self, scenes: str
    # ) -> Dict[str, Any]:
    #     config = self.CONFIG.clone()
    #     config.DATASET.DATA_PATH = scenes
    #     if torch.cuda.device_count() > 0:
    #         self.GPU_ID = (self.GPU_ID + 1) % 7
    #     else:
    #         self.GPU_ID = -1
    #     config.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = self.GPU_ID + 1
    #     return {
    #         "env_config": config,
    #         "max_steps": self.MAX_STEPS,
    #         "sensors": self.SENSORS,
    #         "action_space": gym.spaces.Discrete(len(PointNavTask.action_names())),
    #         "distance_to_goal": self.DISTANCE_TO_GOAL
    #     }

    def get_train_configs(self):
        if self._train_configs is None:
            self._train_configs = construct_env_configs(self.CONFIG)
        return self._train_configs

    def train_task_sampler_args(
        self,
        process_ind: int,
        total_processes: int,
        devices: Optional[List[int]],
        seeds: Optional[List[int]] = None,
        deterministic_cudnn: bool = False,
    ) -> Dict[str, Any]:
        print("Process ind:", process_ind)
        config = self.TRAIN_CONFIGS[process_ind]
        return {
            "env_config": config,
            "max_steps": self.MAX_STEPS,
            "sensors": self.SENSORS,
            "action_space": gym.spaces.Discrete(len(PointNavTask.action_names())),
            "distance_to_goal": self.DISTANCE_TO_GOAL,
            "max_tasks": 4931496  # number of train episodes in gibson
        }

    def valid_task_sampler_args(
        self, process_ind: int,
        total_processes: int,
        devices: Optional[List[int]],
        seeds: Optional[List[int]] = None,
        deterministic_cudnn: bool = False,
    ) -> Dict[str, Any]:
        config = self.CONFIG.clone()
        config.DATASET.DATA_PATH = self.VALID_SCENES
        return {
            "env_config": config,
            "max_steps": self.MAX_STEPS,
            "sensors": self.SENSORS,
            "action_space": gym.spaces.Discrete(len(PointNavTask.action_names())),
            "distance_to_goal": self.DISTANCE_TO_GOAL,
            "max_tasks": 994  # Val mini is only 30 tasks
        }

    def test_task_sampler_args(
        self, process_ind: int,
        total_processes: int,
        devices: Optional[List[int]],
        seeds: Optional[List[int]] = None,
        deterministic_cudnn: bool = False,
    ) -> Dict[str, Any]:
        config = self.CONFIG.clone()
        config.DATASET.DATA_PATH = self.TEST_SCENES
        return {
            "env_config": config,
            "max_steps": self.MAX_STEPS,
            "sensors": self.SENSORS,
            "action_space": gym.spaces.Discrete(len(PointNavTask.action_names())),
            "distance_to_goal": self.DISTANCE_TO_GOAL,
            "max_tasks": 994  # Val mini is only 30 tasks
        }
