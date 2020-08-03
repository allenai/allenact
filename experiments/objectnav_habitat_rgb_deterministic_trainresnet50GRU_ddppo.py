import gym
import torch.nn as nn
from torchvision import models

from projects.objectnav_baselines.models.object_nav_models import ObjectNavActorCriticTrainResNet50GRU
from rl_habitat.habitat_tasks import ObjectNavTask
from rl_habitat.habitat_sensors import RGBSensorHabitat, TargetObjectSensorHabitat
from rl_habitat.habitat_utils import construct_env_configs
from rl_habitat.habitat_preprocessors import ResnetPreProcessorHabitat
from experiments.objectnav_habitat_base_ddppo import ObjectNavHabitatDDPPOBaseExperimentConfig


class ObjectNavHabitatRGBDeterministicTrainResNet50GRUDDPPOExperimentConfig(ObjectNavHabitatDDPPOBaseExperimentConfig):
    """A Point Navigation experiment configuraqtion in Habitat"""

    SENSORS = [
        RGBSensorHabitat(
            {
                "height": ObjectNavHabitatDDPPOBaseExperimentConfig.SCREEN_SIZE,
                "width": ObjectNavHabitatDDPPOBaseExperimentConfig.SCREEN_SIZE,
                "use_resnet_normalization": True,
            }
        ),
        TargetObjectSensorHabitat({}),
    ]

    PREPROCESSORS = [
        ResnetPreProcessorHabitat(
            config={
                "input_height": ObjectNavHabitatDDPPOBaseExperimentConfig.SCREEN_SIZE,
                "input_width": ObjectNavHabitatDDPPOBaseExperimentConfig.SCREEN_SIZE,
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

    CONFIG = ObjectNavHabitatDDPPOBaseExperimentConfig.CONFIG.clone()
    CONFIG.SIMULATOR.AGENT_0.SENSORS = ['RGB_SENSOR']
    CONFIG.SIMULATOR.RGB_SENSOR.WIDTH = ObjectNavHabitatDDPPOBaseExperimentConfig.SCREEN_SIZE
    CONFIG.SIMULATOR.RGB_SENSOR.HEIGHT = ObjectNavHabitatDDPPOBaseExperimentConfig.SCREEN_SIZE
    CONFIG.SIMULATOR.RGB_SENSOR.POSITION = [0, 0.88, 0]

    TRAIN_CONFIGS = construct_env_configs(CONFIG)

    @classmethod
    def create_model(cls, **kwargs) -> nn.Module:
        return ObjectNavActorCriticTrainResNet50GRU(
            action_space=gym.spaces.Discrete(len(ObjectNavTask.action_names())),
            observation_space=kwargs["observation_set"].observation_spaces,
            goal_sensor_uuid="target_object_id",
            hidden_size=512,
            object_type_embedding_dim=32,
            trainable_masked_hidden_state=False,
            num_rnn_layers=1,
            rnn_type='GRU'
        )
