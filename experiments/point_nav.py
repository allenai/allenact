from typing import Dict, Any, List

import gym
import torch
import torch.nn as nn
import torch.optim as optim

from configs.losses import algo_defaults
from configs.util import Builder
from models.point_nav_models import PointNavBaselineActorCritic
from onpolicy_sync.losses import PPO
from rl_base.experiment_config import ExperimentConfig
from rl_base.sensor import SensorSuite, ExpertActionSensor
from rl_base.task import TaskSampler
import habitat
from extensions.habitat.tasks import PointNavTask
from extensions.habitat.task_samplers import PointNavTaskSampler
from extensions.habitat.sensors import RGBSensorHabitat, TargetCoordinatesSensorHabitat


class PointNavHabitatGibsonExperimentConfig(ExperimentConfig):
    """A Point Navigation experiment configuraqtion in Habitat"""

    TRAIN_SCENES = "../habitat-api/data/datasets/pointnav/habitat-test-scenes/v1/train/train.json.gz"
    VALID_SCENES = "../habitat-api/data/datasets/pointnav/habitat-test-scenes/v1/val/val.json.gz"
    TEST_SCENES = "../habitat-api/data/datasets/pointnav/habitat-test-scenes/v1/test/test.json.gz"

    SCREEN_SIZE = 224
    MAX_STEPS = 128

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

    CONFIG = habitat.get_config()
    CONFIG.defrost()
    CONFIG.DATASET.SCENES_DIR = 'habitat/habitat-api/data/scene_datasets/'
    CONFIG.SIMULATOR.AGENT_0.SENSORS = ['RGB_SENSOR']
    CONFIG.SIMULATOR.SEMANTIC_SENSOR.WIDTH = SCREEN_SIZE
    CONFIG.SIMULATOR.SEMANTIC_SENSOR.HEIGHT = SCREEN_SIZE
    CONFIG.SIMULATOR.TURN_ANGLE = 90

    @classmethod
    def tag(cls):
        return "PointNav"

    @classmethod
    def training_pipeline(cls, **kwargs):
        ppo_steps = 1e6
        nprocesses = 4
        lr = 2.5e-4
        num_mini_batch = 1
        update_repeats = 2
        num_steps = 16
        save_interval = 100
        log_interval = 2 * num_steps * nprocesses
        gpu_ids = None if not torch.cuda.is_available() else [0]
        gamma = 0.99
        use_gae = True
        gae_lambda = 1.0
        max_grad_norm = 0.5
        return {
            "save_interval": save_interval,
            "log_interval": log_interval,
            "optimizer": Builder(optim.Adam, dict(lr=lr)),
            "nprocesses": nprocesses,
            "num_mini_batch": num_mini_batch,
            "update_repeats": update_repeats,
            "num_steps": num_steps,
            "gpu_ids": gpu_ids,
            "ppo_loss": Builder(PPO, dict(), default=algo_defaults["ppo_loss"]),
            "gamma": gamma,
            "use_gae": use_gae,
            "gae_lambda": gae_lambda,
            "max_grad_norm": max_grad_norm,
            "pipeline": [
                {
                    "losses": ["ppo_loss"],
                    "end_criterion": ppo_steps
                }
            ],
        }

    @classmethod
    def create_model(cls, **kwargs) -> nn.Module:
        return PointNavBaselineActorCritic(
            action_space=gym.spaces.Discrete(len(PointNavTask.action_names())),
            observation_space=SensorSuite(cls.SENSORS).observation_spaces,
            goal_sensor_uuid="target_coordinates_ind",
            hidden_size=512,
            embed_coordinates=True,
            coordinate_dims=2,
        )

    @classmethod
    def make_sampler_fn(cls, **kwargs) -> TaskSampler:
        return PointNavTaskSampler(**kwargs)

    def _get_sampler_args(
        self, scenes: str
    ) -> Dict[str, Any]:
        self.CONFIG.DATASET.DATA_PATH = scenes
        return {
            "env_config": self.CONFIG,
            "max_steps": self.MAX_STEPS,
            "sensors": self.SENSORS,
            "action_space": gym.spaces.Discrete(len(PointNavTask.action_names())),
        }

    def train_task_sampler_args(
        self, process_ind: int, total_processes: int
    ) -> Dict[str, Any]:
        return self._get_sampler_args(self.TRAIN_SCENES)

    def valid_task_sampler_args(
        self, process_ind: int, total_processes: int
    ) -> Dict[str, Any]:
        return self._get_sampler_args(self.VALID_SCENES)

    def test_task_sampler_args(
        self, process_ind: int, total_processes: int
    ) -> Dict[str, Any]:
        return self._get_sampler_args(self.TEST_SCENES)
