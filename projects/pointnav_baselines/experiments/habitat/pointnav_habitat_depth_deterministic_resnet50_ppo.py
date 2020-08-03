import gym
import habitat
import torch.nn as nn
from torchvision import models

<<<<<<< HEAD:projects/pointnav_baselines/experiments/habitat/pointnav_habitat_depth_deterministic_resnet50_ppo.py
from projects.pointnav_baselines.models.point_nav_models import PointNavActorCriticResNet50
=======
from experiments.pointnav_habitat_base import PointNavHabitatBaseExperimentConfig
from models.point_nav_models import PointNavActorCriticResNet50
>>>>>>> de752be39b1a7d9a4e4dc432293e3a12387385d2:experiments/pointnav_habitat_depth_deterministic_resnet50_ppo.py
from rl_base.sensor import SensorSuite
from rl_habitat.habitat_preprocessors import ResnetPreProcessorHabitat
from rl_habitat.habitat_sensors import (
    DepthSensorHabitat,
    TargetCoordinatesSensorHabitat,
)
from rl_habitat.habitat_tasks import PointNavTask
from rl_habitat.habitat_utils import construct_env_configs
<<<<<<< HEAD:projects/pointnav_baselines/experiments/habitat/pointnav_habitat_depth_deterministic_resnet50_ppo.py
from projects.pointnav_baselines.experiments.habitat.pointnav_habitat_base import PointNavHabitatBaseExperimentConfig
=======
>>>>>>> de752be39b1a7d9a4e4dc432293e3a12387385d2:experiments/pointnav_habitat_depth_deterministic_resnet50_ppo.py


class PointNavHabitatDepthDeterministicResNet50PPOExperimentConfig(
    PointNavHabitatBaseExperimentConfig
):
    """A Point Navigation experiment configuraqtion in Habitat."""

    SENSORS = [
        DepthSensorHabitat(
            **{
                "height": PointNavHabitatBaseExperimentConfig.SCREEN_SIZE,
                "width": PointNavHabitatBaseExperimentConfig.SCREEN_SIZE,
                "use_resnet_normalization": True,
            }
        ),
        TargetCoordinatesSensorHabitat(**{"coordinate_dims": 2}),
    ]

    PREPROCESSORS = [
        ResnetPreProcessorHabitat(
            **{
                "input_height": PointNavHabitatBaseExperimentConfig.SCREEN_SIZE,
                "input_width": PointNavHabitatBaseExperimentConfig.SCREEN_SIZE,
                "output_width": 1,
                "output_height": 1,
                "output_dims": 2048,
                "pool": True,
                "torchvision_resnet_model": models.resnet50,
                "input_uuids": ["depth"],
                "output_uuid": "depth_resnet",
            }
        ),
    ]

    OBSERVATIONS = [
        "depth_resnet",
        "target_coordinates_ind",
    ]

    CONFIG = PointNavHabitatBaseExperimentConfig.CONFIG.clone()
    CONFIG.SIMULATOR.AGENT_0.SENSORS = ["DEPTH_SENSOR"]

    TRAIN_CONFIGS = construct_env_configs(CONFIG)

    TEST_CONFIG = CONFIG.clone()
    TEST_CONFIG.defrost()
    TEST_CONFIG.DATASET.DATA_PATH = PointNavHabitatBaseExperimentConfig.VALID_SCENES
    TEST_CONFIG.freeze()
    TEST_CONFIGS = construct_env_configs(TEST_CONFIG)

    @classmethod
    def train_config(cls, process_ind: int) -> habitat.Config:
        return cls.TRAIN_CONFIGS[process_ind]

    @classmethod
    def create_model(cls, **kwargs) -> nn.Module:
        return PointNavActorCriticResNet50(
            action_space=gym.spaces.Discrete(len(PointNavTask.class_action_names())),
            observation_space=SensorSuite(cls.SENSORS).observation_spaces,
            goal_sensor_uuid="target_coordinates_ind",
            hidden_size=512,
            embed_coordinates=False,
            coordinate_dims=2,
        )
