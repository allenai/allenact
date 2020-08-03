import gym
import torch.nn as nn

from projects.pointnav_baselines.models.point_nav_models import PointNavActorCriticSimpleConvLSTM
from rl_habitat.habitat_tasks import PointNavTask
from rl_habitat.habitat_sensors import DepthSensorHabitat, TargetCoordinatesSensorHabitat
from rl_habitat.habitat_utils import construct_env_configs
from projects.pointnav_baselines.experiments.habitat.pointnav_habitat_base_ddppo import PointNavHabitatDDPPOBaseExperimentConfig


class PointNavHabitatDepthDeterministiSimpleConvGRUDDPPOExperimentConfig(PointNavHabitatDDPPOBaseExperimentConfig):
    """A Point Navigation experiment configuraqtion in Habitat"""

    SENSORS = [
        DepthSensorHabitat(
            {
                "height": PointNavHabitatDDPPOBaseExperimentConfig.SCREEN_SIZE,
                "width": PointNavHabitatDDPPOBaseExperimentConfig.SCREEN_SIZE,
                "use_resnet_normalization": True,
            }
        ),
        TargetCoordinatesSensorHabitat({"coordinate_dims": 2}),
    ]

    PREPROCESSORS = []

    OBSERVATIONS = [
        "depth",
        "target_coordinates_ind",
    ]

    CONFIG = PointNavHabitatDDPPOBaseExperimentConfig.CONFIG.clone()
    CONFIG.SIMULATOR.AGENT_0.SENSORS = ['DEPTH_SENSOR']

    TRAIN_CONFIGS = construct_env_configs(CONFIG)

    @classmethod
    def create_model(cls, **kwargs) -> nn.Module:
        return PointNavActorCriticSimpleConvLSTM(
            action_space=gym.spaces.Discrete(len(PointNavTask.action_names())),
            observation_space=kwargs["observation_set"].observation_spaces,
            goal_sensor_uuid="target_coordinates_ind",
            hidden_size=512,
            embed_coordinates=False,
            coordinate_dims=2,
            num_rnn_layers=1,
            rnn_type='GRU'
        )
