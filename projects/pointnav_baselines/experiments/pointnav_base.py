from abc import ABC
from typing import Optional

from core.base_abstractions.experiment_config import ExperimentConfig


class PointNavBaseConfig(ExperimentConfig, ABC):
    """An Object Navigation experiment configuration in iThor."""

    CAMERA_WIDTH = 400
    CAMERA_HEIGHT = 300
    SCREEN_SIZE = 224
    MAX_STEPS = 500
    ADVANCE_SCENE_ROLLOUT_PERIOD: Optional[int] = None
    STEP_SIZE = 0.25
    ROTATION_DEGREES = 30.0
    DISTANCE_TO_GOAL = 0.2
    STOCHASTIC = True

    def __init__(self):
        self.REWARD_CONFIG = {
            "step_penalty": -0.01,
            "goal_success_reward": 10.0,
            "failed_stop_reward": 0.0,
            "reached_max_steps_reward": 0.0,
            "shaping_weight": 1.0,
        }
