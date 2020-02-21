import gym
import torch.nn as nn

import habitat
from models.point_nav_models import PointNavActorCriticSimpleConv
from rl_base.sensor import SensorSuite
from rl_habitat.habitat_tasks import PointNavTask
from rl_habitat.habitat_sensors import RGBSensorHabitat, TargetCoordinatesSensorHabitat
from rl_habitat.habitat_utils import construct_env_configs
from experiments.pointnav_habitat_base import PointNavHabitatBaseExperimentConfig


class PointNavHabitatRGBDeterministicResNet50GRUPPOExperimentConfig(PointNavHabitatBaseExperimentConfig):
    """A Point Navigation experiment configuraqtion in Habitat"""

    SENSORS = [
        RGBSensorHabitat(
            {
                "height": SCREEN_SIZE,
                "width": SCREEN_SIZE,
                "use_resnet_normalization": True,
            }
        ),
        TargetCoordinatesSensorHabitat({"coordinate_dims": 2}),
    ]

    OBSERVATIONS = [
        "rgb",
        "target_coordinates_ind",
    ]

    PREPROCESSORS = []

    CONFIG = habitat.get_config('configs/gibson.yaml')
    CONFIG.defrost()
    CONFIG.NUM_PROCESSES = NUM_PROCESSES
    CONFIG.SIMULATOR_GPU_IDS = [1]
    CONFIG.DATASET.SCENES_DIR = 'habitat/habitat-api/data/scene_datasets/'
    CONFIG.DATASET.POINTNAVV1.CONTENT_SCENES = ['*']
    CONFIG.DATASET.DATA_PATH = TRAIN_SCENES
    CONFIG.SIMULATOR.AGENT_0.SENSORS = ['RGB_SENSOR']
    CONFIG.SIMULATOR.RGB_SENSOR.WIDTH = SCREEN_SIZE
    CONFIG.SIMULATOR.RGB_SENSOR.HEIGHT = SCREEN_SIZE
    CONFIG.SIMULATOR.TURN_ANGLE = 45
    CONFIG.SIMULATOR.FORWARD_STEP_SIZE = 0.25
    CONFIG.ENVIRONMENT.MAX_EPISODE_STEPS = MAX_STEPS

    CONFIG.TASK.TYPE = 'Nav-v0'
    CONFIG.TASK.SUCCESS_DISTANCE = DISTANCE_TO_GOAL
    CONFIG.TASK.SENSORS = ['POINTGOAL_WITH_GPS_COMPASS_SENSOR']
    CONFIG.TASK.POINTGOAL_WITH_GPS_COMPASS_SENSOR.GOAL_FORMAT = "POLAR"
    CONFIG.TASK.POINTGOAL_WITH_GPS_COMPASS_SENSOR.DIMENSIONALITY = 2
    CONFIG.TASK.GOAL_SENSOR_UUID = 'pointgoal_with_gps_compass'
    CONFIG.TASK.MEASUREMENTS = ['DISTANCE_TO_GOAL', 'SPL']
    CONFIG.TASK.SPL.TYPE = 'SPL'
    CONFIG.TASK.SPL.SUCCESS_DISTANCE = 0.2

    TRAIN_CONFIGS = construct_env_configs(CONFIG)

    @classmethod
    def create_model(cls, **kwargs) -> nn.Module:
        return PointNavActorCriticSimpleConv(
            action_space=gym.spaces.Discrete(len(PointNavTask.action_names())),
            observation_space=SensorSuite(cls.SENSORS).observation_spaces,
            goal_sensor_uuid="target_coordinates_ind",
            hidden_size=512,
            embed_coordinates=False,
            coordinate_dims=2,
        )
