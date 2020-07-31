import gym
import habitat
import torch.nn as nn

from experiments.objectnav_habitat_base import ObjectNavHabitatBaseExperimentConfig
from models.object_nav_models import ObjectNavBaselineActorCritic
from rl_base.sensor import SensorSuite
from rl_habitat.habitat_sensors import DepthSensorHabitat, TargetObjectSensorHabitat
from rl_habitat.habitat_tasks import ObjectNavTask
from rl_habitat.habitat_utils import construct_env_configs


class ObjectNavHabitatDebthDeterministicSimpleConvGRUPPOExperimentConfig(
    ObjectNavHabitatBaseExperimentConfig
):
    """A Point Navigation experiment configuraqtion in Habitat."""

    SENSORS = [
        DepthSensorHabitat(
            **{
                "height": ObjectNavHabitatBaseExperimentConfig.SCREEN_SIZE,
                "width": ObjectNavHabitatBaseExperimentConfig.SCREEN_SIZE,
                "use_resnet_normalization": True,
            }
        ),
        TargetObjectSensorHabitat(),
    ]

    PREPROCESSORS = []

    OBSERVATIONS = [
        "depth",
        "target_object_id",
    ]

    CONFIG = ObjectNavHabitatBaseExperimentConfig.CONFIG.clone()
    CONFIG.SIMULATOR.AGENT_0.SENSORS = ["DEPTH_SENSOR"]
    CONFIG.SIMULATOR.DEPTH_SENSOR.WIDTH = (
        ObjectNavHabitatBaseExperimentConfig.SCREEN_SIZE
    )
    CONFIG.SIMULATOR.DEPTH_SENSOR.HEIGHT = (
        ObjectNavHabitatBaseExperimentConfig.SCREEN_SIZE
    )
    CONFIG.SIMULATOR.DEPTH_SENSOR.POSITION = [0, 0.88, 0]

    TRAIN_CONFIGS = construct_env_configs(CONFIG)

    @classmethod
    def train_config(cls, process_ind: int) -> habitat.Config:
        return cls.TRAIN_CONFIGS[process_ind]

    @classmethod
    def create_model(cls, **kwargs) -> nn.Module:
        return ObjectNavBaselineActorCritic(
            action_space=gym.spaces.Discrete(len(ObjectNavTask.class_action_names())),
            observation_space=SensorSuite(cls.SENSORS).observation_spaces,
            goal_sensor_uuid="target_object_id",
            hidden_size=512,
            object_type_embedding_dim=32,
            num_rnn_layers=1,
            rnn_type="GRU",
        )
