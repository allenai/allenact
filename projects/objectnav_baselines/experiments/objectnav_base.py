from abc import ABC

from core.base_abstractions.experiment_config import ExperimentConfig


class ObjectNavBaseConfig(ExperimentConfig, ABC):
    """The base object navigation configuration file."""

    CAMERA_WIDTH = 400
    CAMERA_HEIGHT = 300
    SCREEN_SIZE = 224
    MAX_STEPS = 500
    STEP_SIZE = 0.25
    ROTATION_DEGREES = 30.0
    VISIBILITY_DISTANCE = 1.0
    STOCHASTIC = True

    def __init__(self):
        self.REWARD_CONFIG = {
            "step_penalty": -0.01,
            "goal_success_reward": 10.0,
            "failed_stop_reward": 0.0,
            "shaping_weight": 1.0,
        }
