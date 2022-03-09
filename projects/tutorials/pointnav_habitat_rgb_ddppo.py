import os
from typing import Dict, Any, List, Optional, Sequence

import gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torchvision import models

from allenact.algorithms.onpolicy_sync.losses import PPO
from allenact.algorithms.onpolicy_sync.losses.ppo import PPOConfig
from allenact.base_abstractions.experiment_config import ExperimentConfig, MachineParams
from allenact.base_abstractions.preprocessor import SensorPreprocessorGraph
from allenact.base_abstractions.sensor import SensorSuite
from allenact.base_abstractions.task import TaskSampler
from allenact.embodiedai.preprocessors.resnet import ResNetPreprocessor
from allenact.utils.experiment_utils import (
    Builder,
    PipelineStage,
    TrainingPipeline,
    LinearDecay,
    evenly_distribute_count_into_bins,
)
from allenact_plugins.habitat_plugin.habitat_constants import (
    HABITAT_DATASETS_DIR,
    HABITAT_CONFIGS_DIR,
)
from allenact_plugins.habitat_plugin.habitat_sensors import (
    RGBSensorHabitat,
    TargetCoordinatesSensorHabitat,
)
from allenact_plugins.habitat_plugin.habitat_task_samplers import PointNavTaskSampler
from allenact_plugins.habitat_plugin.habitat_utils import (
    construct_env_configs,
    get_habitat_config,
)
from allenact_plugins.navigation_plugin.objectnav.models import (
    ResnetTensorNavActorCritic,
)
from allenact_plugins.robothor_plugin.robothor_tasks import PointNavTask


class PointNavHabitatRGBPPOTutorialExperimentConfig(ExperimentConfig):
    """A Point Navigation experiment configuration in Habitat."""

    # Task Parameters
    MAX_STEPS = 500
    REWARD_CONFIG = {
        "step_penalty": -0.01,
        "goal_success_reward": 10.0,
        "failed_stop_reward": 0.0,
        "shaping_weight": 1.0,
    }
    DISTANCE_TO_GOAL = 0.2

    # Simulator Parameters
    CAMERA_WIDTH = 640
    CAMERA_HEIGHT = 480
    SCREEN_SIZE = 224

    # Training Engine Parameters
    ADVANCE_SCENE_ROLLOUT_PERIOD: Optional[int] = None
    NUM_PROCESSES = max(5 * torch.cuda.device_count() - 1, 4)
    TRAINING_GPUS = list(range(torch.cuda.device_count()))
    VALIDATION_GPUS = [torch.cuda.device_count() - 1]
    TESTING_GPUS = [torch.cuda.device_count() - 1]

    task_data_dir_template = os.path.join(
        HABITAT_DATASETS_DIR, "pointnav/gibson/v1/{}/{}.json.gz"
    )
    TRAIN_SCENES = task_data_dir_template.format(*(["train"] * 2))
    VALID_SCENES = task_data_dir_template.format(*(["val"] * 2))
    TEST_SCENES = task_data_dir_template.format(*(["test"] * 2))

    CONFIG = get_habitat_config(
        os.path.join(HABITAT_CONFIGS_DIR, "tasks/pointnav_gibson.yaml")
    )
    CONFIG.defrost()
    CONFIG.NUM_PROCESSES = NUM_PROCESSES
    CONFIG.SIMULATOR_GPU_IDS = TRAINING_GPUS
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
    CONFIG.TASK.SUCCESS_DISTANCE = DISTANCE_TO_GOAL
    CONFIG.TASK.SENSORS = ["POINTGOAL_WITH_GPS_COMPASS_SENSOR"]
    CONFIG.TASK.POINTGOAL_WITH_GPS_COMPASS_SENSOR.GOAL_FORMAT = "POLAR"
    CONFIG.TASK.POINTGOAL_WITH_GPS_COMPASS_SENSOR.DIMENSIONALITY = 2
    CONFIG.TASK.GOAL_SENSOR_UUID = "pointgoal_with_gps_compass"
    CONFIG.TASK.MEASUREMENTS = ["DISTANCE_TO_GOAL", "SUCCESS", "SPL"]
    CONFIG.TASK.SPL.TYPE = "SPL"
    CONFIG.TASK.SPL.SUCCESS_DISTANCE = DISTANCE_TO_GOAL
    CONFIG.TASK.SUCCESS.SUCCESS_DISTANCE = DISTANCE_TO_GOAL

    CONFIG.MODE = "train"

    SENSORS = [
        RGBSensorHabitat(
            height=SCREEN_SIZE, width=SCREEN_SIZE, use_resnet_normalization=True,
        ),
        TargetCoordinatesSensorHabitat(coordinate_dims=2),
    ]

    PREPROCESSORS = [
        Builder(
            ResNetPreprocessor,
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
                else evenly_distribute_count_into_bins(self.NUM_PROCESSES, len(gpu_ids))
            )
        elif mode == "valid":
            nprocesses = 1
            gpu_ids = [] if not torch.cuda.is_available() else self.VALIDATION_GPUS
        elif mode == "test":
            nprocesses = 1
            gpu_ids = [] if not torch.cuda.is_available() else self.TESTING_GPUS
        else:
            raise NotImplementedError("mode must be 'train', 'valid', or 'test'.")

        sensor_preprocessor_graph = (
            SensorPreprocessorGraph(
                source_observation_spaces=SensorSuite(self.SENSORS).observation_spaces,
                preprocessors=self.PREPROCESSORS,
            )
            if mode == "train"
            or (
                (isinstance(nprocesses, int) and nprocesses > 0)
                or (isinstance(nprocesses, Sequence) and sum(nprocesses) > 0)
            )
            else None
        )

        return MachineParams(
            nprocesses=nprocesses,
            devices=gpu_ids,
            sensor_preprocessor_graph=sensor_preprocessor_graph,
        )

    # Define Model
    @classmethod
    def create_model(cls, **kwargs) -> nn.Module:
        return ResnetTensorNavActorCritic(
            action_space=gym.spaces.Discrete(len(PointNavTask.class_action_names())),
            observation_space=kwargs["sensor_preprocessor_graph"].observation_spaces,
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
            "action_space": gym.spaces.Discrete(len(PointNavTask.class_action_names())),
            "distance_to_goal": self.DISTANCE_TO_GOAL,  # type:ignore
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
            "action_space": gym.spaces.Discrete(len(PointNavTask.class_action_names())),
            "distance_to_goal": self.DISTANCE_TO_GOAL,  # type:ignore
        }

    def test_task_sampler_args(
        self,
        process_ind: int,
        total_processes: int,
        devices: Optional[List[int]] = None,
        seeds: Optional[List[int]] = None,
        deterministic_cudnn: bool = False,
    ) -> Dict[str, Any]:
        raise NotImplementedError("Testing not implemented for this tutorial.")
