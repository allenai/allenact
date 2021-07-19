from abc import ABC
from typing import Optional, Sequence, Union

from allenact.base_abstractions.experiment_config import ExperimentConfig
from allenact.base_abstractions.preprocessor import Preprocessor
from allenact.base_abstractions.sensor import Sensor
from allenact.utils.experiment_utils import Builder


class ArmPointNavBaseConfig(ExperimentConfig, ABC):
    """The base object navigation configuration file."""

    ADVANCE_SCENE_ROLLOUT_PERIOD: Optional[int] = None
    SENSORS: Optional[Sequence[Sensor]] = None

    STEP_SIZE = 0.25
    ROTATION_DEGREES = 45.0
    VISIBILITY_DISTANCE = 1.0
    STOCHASTIC = False

    CAMERA_WIDTH = 224
    CAMERA_HEIGHT = 224
    SCREEN_SIZE = 224
    MAX_STEPS = 200

    def __init__(self):
        self.REWARD_CONFIG = {
            "step_penalty": -0.01,
            "goal_success_reward": 10.0,
            "pickup_success_reward": 10.0,
            "failed_stop_reward": 0.0,
            "shaping_weight": 1.0,  # we are not using this
            "failed_action_penalty": -0.03,
        }

    @classmethod
    def preprocessors(cls) -> Sequence[Union[Preprocessor, Builder[Preprocessor]]]:
        return tuple()
