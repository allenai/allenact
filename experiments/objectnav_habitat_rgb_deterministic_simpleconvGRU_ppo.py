import gym
import torch.nn as nn

from models.point_nav_models import PointNavActorCriticSimpleConvLSTM
from rl_base.sensor import SensorSuite
from rl_habitat.habitat_tasks import PointNavTask
from rl_habitat.habitat_sensors import RGBSensorHabitat, TargetObjectSensorHabitat
from rl_habitat.habitat_utils import construct_env_configs
from experiments.objectnav_habitat_base import ObjectNavHabitatBaseExperimentConfig


class ObjectNavHabitatRGBeterministicSimpleConvGRUPPOExperimentConfig(ObjectNavHabitatBaseExperimentConfig):
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

    PREPROCESSORS = []

    OBSERVATIONS = [
        "rgb",
        "target_object_id",
    ]

    CONFIG = ObjectNavHabitatBaseExperimentConfig.CONFIG.clone()
    CONFIG.SIMULATOR.AGENT_0.SENSORS = ['RGB_SENSOR']

    TRAIN_CONFIGS = construct_env_configs(CONFIG)

    @classmethod
    def create_model(cls, **kwargs) -> nn.Module:
        return PointNavActorCriticSimpleConvLSTM(
            action_space=gym.spaces.Discrete(len(PointNavTask.action_names())),
            observation_space=SensorSuite(cls.SENSORS).observation_spaces,
            goal_sensor_uuid="target_object_id",
            hidden_size=512,
            embed_coordinates=False,
            coordinate_dims=2,
            num_rnn_layers=1,
            rnn_type='GRU'
        )
