from abc import ABC
from typing import Optional, Sequence, Union

from core.base_abstractions.experiment_config import ExperimentConfig
from core.base_abstractions.preprocessor import Preprocessor
from core.base_abstractions.sensor import Sensor
from utils.experiment_utils import Builder


class PointNavBaseConfig(ExperimentConfig, ABC):
    """An Object Navigation experiment configuration in iThor."""

    ADVANCE_SCENE_ROLLOUT_PERIOD: Optional[int] = None
    PREPROCESSORS: Sequence[Union[Preprocessor, Builder[Preprocessor]]] = tuple()
    SENSORS: Optional[Sequence[Sensor]] = None

    STEP_SIZE = 0.25
    ROTATION_DEGREES = 30.0
    DISTANCE_TO_GOAL = 0.2
    STOCHASTIC = True

    CAMERA_WIDTH = 400
    CAMERA_HEIGHT = 300
    SCREEN_SIZE = 224
    MAX_STEPS = 500

    def __init__(self):
        self.REWARD_CONFIG = {
            "step_penalty": -0.01,
            "goal_success_reward": 10.0,
            "failed_stop_reward": 0.0,
            "reached_max_steps_reward": 0.0,
            "shaping_weight": 1.0,
        }
