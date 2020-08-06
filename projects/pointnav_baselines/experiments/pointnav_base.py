from common.base_abstractions.experiment_config import ExperimentConfig


class PointNavBaseConfig(ExperimentConfig):
    """An Object Navigation experiment configuration in iThor."""

    def __init__(self):
        self.CAMERA_WIDTH = 640
        self.CAMERA_HEIGHT = 480
        self.SCREEN_SIZE = 224
        self.MAX_STEPS = 500
        self.ADVANCE_SCENE_ROLLOUT_PERIOD = 10000000000000
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
