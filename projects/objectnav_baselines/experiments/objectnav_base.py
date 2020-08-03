from typing import Dict, Any, List, Optional
from math import ceil

import gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torchvision import models
import numpy as np
import glob

from onpolicy_sync.losses.ppo import PPOConfig
from projects.objectnav_baselines.models.object_nav_models import ResnetTensorObjectNavActorCritic
from onpolicy_sync.losses import PPO
from rl_base.experiment_config import ExperimentConfig
from rl_base.task import TaskSampler
from rl_base.preprocessor import ObservationSet
from rl_robothor.robothor_tasks import ObjectNavTask
from rl_robothor.robothor_task_samplers import ObjectNavDatasetTaskSampler
from rl_ai2thor.ai2thor_sensors import RGBSensorThor, GoalObjectTypeThorSensor
from rl_habitat.habitat_preprocessors import ResnetPreProcessorHabitat
from utils.experiment_utils import Builder, PipelineStage, TrainingPipeline, LinearDecay


class ObjectNavBaseConfig(ExperimentConfig):
    """An Object Navigation experiment configuration in iThor"""

    def __init__(self):
        self.CAMERA_WIDTH = 640
        self.CAMERA_HEIGHT = 480
        self.SCREEN_SIZE = 224
        self.MAX_STEPS = 500
        self.ADVANCE_SCENE_ROLLOUT_PERIOD = 10000000000000
        self.NUM_PROCESSES = 80
        self.STEP_SIZE = 0.25
        self.ROTATION_DEGREES = 30.0
        self.VISIBILITY_DISTANCE = 1.0
        self.STOCHASTIC = True
        self.REWARD_CONFIG = {
            "step_penalty": -0.01,
            "goal_success_reward": 10.0,
            "failed_stop_reward": 0.0,
            "shaping_weight": 1.0,
        }
