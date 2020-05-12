import gym
import torch.nn as nn
from torchvision import models

from models.object_nav_models import ResnetTensorObjectNavActorCritic
from rl_base.sensor import SensorSuite
from rl_habitat.habitat_tasks import ObjectNavTask
from rl_habitat.habitat_sensors import RGBSensorHabitat, TargetObjectSensorHabitat, AgentCoordinatesSensorHabitat
from rl_habitat.habitat_utils import construct_env_configs, construct_env_configs_mp3d
from rl_habitat.habitat_preprocessors import ResnetPreProcessorHabitat
from rl_base.preprocessor import ObservationSet
from experiments.objectnav_habitat_base_ddppo import ObjectNavHabitatDDPPOBaseExperimentConfig
from utils.experiment_utils import Builder


class ObjectNavHabitatRGBDeterministicTrainResNetJordiGRUDDPPOExperimentConfig(ObjectNavHabitatDDPPOBaseExperimentConfig):
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
        AgentCoordinatesSensorHabitat({})
    ]

    # PREPROCESSORS = [
    #     ResnetPreProcessorHabitat(
    #         config={
    #             "input_height": ObjectNavHabitatDDPPOBaseExperimentConfig.SCREEN_SIZE,
    #             "input_width": ObjectNavHabitatDDPPOBaseExperimentConfig.SCREEN_SIZE,
    #             "output_width": 1,
    #             "output_height": 1,
    #             "output_dims": 2048,
    #             "pool": False,
    #             "torchvision_resnet_model": models.resnet50,
    #             "input_uuids": ["rgb"],
    #             "output_uuid": "rgb_resnet",
    #             "parallel": False,
    #         }
    #     ),
    # ]

    PREPROCESSORS = [
        Builder(ResnetPreProcessorHabitat,
                dict(config={
                    "input_height": ObjectNavHabitatDDPPOBaseExperimentConfig.SCREEN_SIZE,
                    "input_width": ObjectNavHabitatDDPPOBaseExperimentConfig.SCREEN_SIZE,
                    "output_width": 20,  # 8,
                    "output_height": 15,  # 8,
                    "output_dims": 512,
                    "pool": False,
                    "torchvision_resnet_model": models.resnet18,
                    "input_uuids": ["rgb"],
                    "output_uuid": "rgb_resnet",
                    "parallel": False,  # TODO False for debugging
            })
        ),
    ]

    OBSERVATIONS = [
        "rgb_resnet",
        "target_object_id",
    ]

    CONFIG = ObjectNavHabitatDDPPOBaseExperimentConfig.CONFIG.clone()
    CONFIG.SIMULATOR.AGENT_0.SENSORS = ['RGB_SENSOR']
    CONFIG.SIMULATOR.RGB_SENSOR.WIDTH = 640  # ObjectNavHabitatDDPPOBaseExperimentConfig.SCREEN_SIZE
    CONFIG.SIMULATOR.RGB_SENSOR.HEIGHT = 480  # ObjectNavHabitatDDPPOBaseExperimentConfig.SCREEN_SIZE
    CONFIG.SIMULATOR.RGB_SENSOR.POSITION = [0, 0.88, 0]

    TRAIN_CONFIGS = construct_env_configs(CONFIG)

    @classmethod
    def create_model(cls, **kwargs) -> nn.Module:
        return ResnetTensorObjectNavActorCritic(
            action_space=gym.spaces.Discrete(len(ObjectNavTask.action_names())),
            observation_space=kwargs["observation_set"].observation_spaces,
            goal_sensor_uuid="target_object_id",
            resnet_preprocessor_uuid="rgb_resnet",
            hidden_size=512,
            goal_dims=32,
            resnet_compressor_hidden_out_dims=(128, 32),
            combiner_hidden_out_dims=(128, 32)
        )
