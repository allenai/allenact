from typing import Dict, Any, List

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import copy

from configs.losses import PPOConfig
from configs.util import Builder, recursive_update
from extensions.ai2thor.models.object_nav_models import ObjectNavBaselineActorCritic
from extensions.ai2thor.sensors import RGBSensorThor, GoalObjectTypeThorSensor
from extensions.ai2thor.preprocessors import ResnetPreProcessorThor
from extensions.ai2thor.task_samplers import ObjectNavTaskSampler
from extensions.ai2thor.tasks import ObjectNavTask
from onpolicy_sync.utils import LinearDecay
from onpolicy_sync.losses import PPO
from onpolicy_sync.losses.imitation import Imitation
from rl_base.experiment_config import ExperimentConfig
from rl_base.sensor import SensorSuite, ExpertActionSensor
from rl_base.task import TaskSampler
from experiments.object_nav_thor import ObjectNavThorExperimentConfig as Base


class ObjectNavThorPreResnetExperimentConfig(Base):
    """An object navigation experiment in THOR."""

    OBJECT_TYPES = sorted(["Cup", "Television", "Tomato"])

    GPUS = None if not torch.cuda.is_available() else [0]

    SENSORS = [
        RGBSensorThor(
            {
                "height": Base.SCREEN_SIZE,
                "width": Base.SCREEN_SIZE,
                "use_resnet_normalization": True,
            }
        ),
        GoalObjectTypeThorSensor({"object_types": OBJECT_TYPES}),
        ExpertActionSensor({"nactions": 6}),
    ]

    OBSERVATIONS = [
        # Builder(
        #     ResnetPreProcessorThor,
        #     {
        #         "config": {
        #             "input_height": Base.SCREEN_SIZE,
        #             "input_width": Base.SCREEN_SIZE,
        #             "output_width": 7,
        #             "output_height": 7,
        #             "output_dims": 512,
        #             "torchvision_resnet_model": models.resnet18,
        #             "input_uuids": ["rgb"],
        #             "output_uuid": "resnet",
        #         }
        #     },
        # ),
        "goal_object_type_ind",
        "rgb",
    ]

    ENV_ARGS = {
        "player_screen_height": Base.SCREEN_SIZE,
        "player_screen_width": Base.SCREEN_SIZE,
        "quality": "Very Low",
    }

    MAX_STEPS = 4

    @classmethod
    def tag(cls):
        return "ObjectNavPreResnet"

    @classmethod
    def training_pipeline(cls, **kwargs):
        inherited = super().training_pipeline()
        ppo_steps = 1e6
        nprocesses = 2
        lr = 2.5e-4
        num_mini_batch = 2
        update_repeats = 2
        num_steps = 4
        gpu_ids = cls.GPUS
        override = {
            "optimizer": Builder(optim.Adam, dict(lr=lr)),
            "nprocesses": nprocesses,
            "num_mini_batch": num_mini_batch,
            "update_repeats": update_repeats,
            "num_steps": num_steps,
            "gpu_ids": gpu_ids,
            "imitation_loss": Builder(Imitation,),
            "ppo_loss": Builder(PPO, dict(), default=PPOConfig,),
            "observation_set": cls.OBSERVATIONS,
            "pipeline": [{"losses": ["ppo_loss"], "end_criterion": ppo_steps},],
        }
        return recursive_update(inherited, override)

    @classmethod
    def create_model(cls, **kwargs) -> nn.Module:
        return ObjectNavBaselineActorCritic(
            action_space=gym.spaces.Discrete(len(ObjectNavTask.action_names())),
            observation_space=SensorSuite(cls.SENSORS).observation_spaces,
            goal_sensor_uuid="goal_object_type_ind",
            hidden_size=512,
            object_type_embedding_dim=8,
        )
