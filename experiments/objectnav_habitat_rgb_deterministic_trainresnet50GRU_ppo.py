import gym
import torch.nn as nn
from torchvision import models

from models.object_nav_models import ObjectNavNavActorCriticTrainResNet50GRU
from rl_base.sensor import SensorSuite
from rl_habitat.habitat_tasks import ObjectNavTask
from rl_habitat.habitat_sensors import RGBSensorHabitat, TargetObjectSensorHabitat
from rl_habitat.habitat_utils import construct_env_configs
from rl_habitat.habitat_preprocessors import ResnetPreProcessorHabitat
from experiments.objectnav_habitat_base import ObjectNavHabitatBaseExperimentConfig


class ObjectNavHabitatRGBDeterministicTrainResNet50GRUPPOExperimentConfig(ObjectNavHabitatBaseExperimentConfig):
    """A Point Navigation experiment configuraqtion in Habitat"""

    SENSORS = [
        RGBSensorHabitat(
            {
                "height": ObjectNavHabitatBaseExperimentConfig.SCREEN_SIZE,
                "width": ObjectNavHabitatBaseExperimentConfig.SCREEN_SIZE,
                "use_resnet_normalization": True,
            }
        ),
        TargetObjectSensorHabitat({}),
    ]

    PREPROCESSORS = [
        ResnetPreProcessorHabitat(
            config={
                "input_height": ObjectNavHabitatBaseExperimentConfig.SCREEN_SIZE,
                "input_width": ObjectNavHabitatBaseExperimentConfig.SCREEN_SIZE,
                "output_width": 1,
                "output_height": 1,
                "output_dims": 2048,
                "pool": False,
                "torchvision_resnet_model": models.resnet50,
                "input_uuids": ["rgb"],
                "output_uuid": "rgb_resnet",
                "parallel": False,
            }
        ),
    ]

    OBSERVATIONS = [
        "rgb_resnet",
        "target_object_id",
    ]

    CONFIG = ObjectNavHabitatBaseExperimentConfig.CONFIG.clone()
    CONFIG.SIMULATOR.AGENT_0.SENSORS = ['RGB_SENSOR']
    CONFIG.SIMULATOR.RGB_SENSOR.WIDTH = ObjectNavHabitatBaseExperimentConfig.SCREEN_SIZE
    CONFIG.SIMULATOR.RGB_SENSOR.HEIGHT = ObjectNavHabitatBaseExperimentConfig.SCREEN_SIZE
    CONFIG.SIMULATOR.RGB_SENSOR.POSITION = [0, 0.88, 0]

    TRAIN_CONFIGS = construct_env_configs(CONFIG)

    @classmethod
    def create_model(cls, **kwargs) -> nn.Module:
        return ObjectNavNavActorCriticTrainResNet50GRU(
            action_space=gym.spaces.Discrete(len(ObjectNavTask.action_names())),
            observation_space=SensorSuite(cls.SENSORS).observation_spaces,
            goal_sensor_uuid="target_object_id",
            hidden_size=512,
            object_type_embedding_dim=32,
            trainable_masked_hidden_state = False,
            num_rnn_layers=1,
            rnn_type='GRU'
        )
