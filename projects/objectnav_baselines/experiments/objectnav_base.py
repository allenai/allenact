from abc import ABC
from typing import Optional, Sequence, Union

from core.base_abstractions.experiment_config import ExperimentConfig
from core.base_abstractions.preprocessor import Preprocessor
from core.base_abstractions.sensor import Sensor
from utils.experiment_utils import Builder


class ObjectNavBaseConfig(ExperimentConfig, ABC):
    """The base object navigation configuration file."""

    ADVANCE_SCENE_ROLLOUT_PERIOD: Optional[int] = None
    SENSORS: Optional[Sequence[Sensor]] = None

    STEP_SIZE = 0.25
    ROTATION_DEGREES = 30.0
    VISIBILITY_DISTANCE = 1.0
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
            "shaping_weight": 1.0,
        }

    @classmethod
    def preprocessors(cls) -> Sequence[Union[Preprocessor, Builder[Preprocessor]]]:
        return tuple()
