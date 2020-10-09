from abc import ABC
from typing import Optional

from core.base_abstractions.experiment_config import ExperimentConfig


class PointNavBaseConfig(ExperimentConfig, ABC):
    """An Object Navigation experiment configuration in iThor."""

    def __init__(self):
        self.CAMERA_WIDTH = 400
        self.CAMERA_HEIGHT = 300
        self.SCREEN_SIZE = 224
        self.MAX_STEPS = 500
        self.ADVANCE_SCENE_ROLLOUT_PERIOD: Optional[int] = None
        self.STEP_SIZE = 0.25
        self.ROTATION_DEGREES = 30.0
        self.DISTANCE_TO_GOAL = 0.2
        self.STOCHASTIC = True
        self.REWARD_CONFIG = {
            "step_penalty": -0.01,
            "goal_success_reward": 10.0,
            "failed_stop_reward": 0.0,
            "shaping_weight": 1.0,
        }
