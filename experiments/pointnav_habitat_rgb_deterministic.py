from typing import Dict, Any, List

import gym
import torch
import torch.nn as nn
import torch.optim as optim

from onpolicy_sync.losses.ppo import PPOConfig
from models.point_nav_models import PointNavBaselineActorCritic
from onpolicy_sync.losses import PPO
from rl_base.experiment_config import ExperimentConfig
from rl_base.sensor import SensorSuite, ExpertActionSensor
from rl_base.task import TaskSampler
import habitat
from extensions.habitat.tasks import PointNavTask
from extensions.habitat.task_samplers import PointNavTaskSampler
from extensions.habitat.sensors import RGBSensorHabitat, TargetCoordinatesSensorHabitat
from utils.experiment_utils import Builder, PipelineStage, TrainingPipeline


class PointNavHabitatRGBDeterministicExperimentConfig(ExperimentConfig):
    """A Point Navigation experiment configuraqtion in Habitat"""

    TRAIN_SCENES = "habitat/habitat-api/data/datasets/pointnav/gibson/v1/train/train.json.gz"
    VALID_SCENES = "habitat/habitat-api/data/datasets/pointnav/gibson/v1/val/val.json.gz"
    TEST_SCENES = "habitat/habitat-api/data/datasets/pointnav/gibson/v1/test/test.json.gz"

    SCREEN_SIZE = 256
    MAX_STEPS = 500
    DISTANCE_TO_GOAL = 0.2

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

    CONFIG = habitat.get_config('gibson.yaml')
    CONFIG.defrost()
    CONFIG.DATASET.SCENES_DIR = 'habitat/habitat-api/data/scene_datasets/'
    CONFIG.DATASET.POINTNAVV1.CONTENT_SCENES = ['*']
    CONFIG.SIMULATOR.AGENT_0.SENSORS = ['RGB_SENSOR']
    CONFIG.SIMULATOR.RGB_SENSOR.WIDTH = SCREEN_SIZE
    CONFIG.SIMULATOR.RGB_SENSOR.HEIGHT = SCREEN_SIZE
    CONFIG.SIMULATOR.TURN_ANGLE = 45
    CONFIG.SIMULATOR.FORWARD_STEP_SIZE = 0.25

    GPU_ID = 0


    @classmethod
    def tag(cls):
        return "PointNav"

    @classmethod
    def training_pipeline(cls, **kwargs):
        ppo_steps = 7.5e7
        nprocesses = 8
        lr = 2.5e-4
        num_mini_batch = 1
        update_repeats = 4
        num_steps = 128
        save_interval = 1000000
        log_interval = 2 * num_steps * nprocesses
        gamma = 0.99
        use_gae = True
        gae_lambda = 1.0
        max_grad_norm = 0.5
        return TrainingPipeline(
            save_interval=save_interval,
            log_interval=log_interval,
            optimizer=Builder(optim.Adam, dict(lr=lr)),
            num_mini_batch=num_mini_batch,
            update_repeats=update_repeats,
            num_steps=num_steps,
            named_losses={"ppo_loss": Builder(PPO, dict(), default=PPOConfig,)},
            gamma=gamma,
            use_gae=use_gae,
            gae_lambda=gae_lambda,
            max_grad_norm=max_grad_norm,
            pipeline_stages=[
                PipelineStage(loss_names=["ppo_loss"], end_criterion=ppo_steps,),
            ],
            scheduler=Builder(
                torch.optim.lr_scheduler.LambdaLR,
                dict(lr_lambda=(lambda num_updates: (lambda x: 1 - (x / float(num_updates))))(ppo_steps))
            )
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

    @classmethod
    def machine_params(cls, mode="train", **kwargs):
        if mode == "train":
            nprocesses = 1 if not torch.cuda.is_available() else 6
            gpu_ids = [] if not torch.cuda.is_available() else [0]
        elif mode == "valid":
            nprocesses = 1
            gpu_ids = [] if not torch.cuda.is_available() else [1]
        elif mode == "test":
            nprocesses = 1
            gpu_ids = [] if not torch.cuda.is_available() else [0]
        else:
            raise NotImplementedError("mode must be 'train', 'valid', or 'test'.")

        return {"nprocesses": nprocesses, "gpu_ids": gpu_ids}

    @classmethod
    def create_model(cls, **kwargs) -> nn.Module:
        return PointNavBaselineActorCritic(
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

    def _get_sampler_args(
        self, scenes: str
    ) -> Dict[str, Any]:
        config = self.CONFIG.clone()
        config.DATASET.DATA_PATH = scenes
        self.GPU_ID = (self.GPU_ID + 1) % torch.cuda.device_count()
        config.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = self.GPU_ID
        return {
            "env_config": config,
            "max_steps": self.MAX_STEPS,
            "sensors": self.SENSORS,
            "action_space": gym.spaces.Discrete(len(PointNavTask.action_names())),
            "distance_to_goal": self.DISTANCE_TO_GOAL
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
