from typing import Sequence, Union

import gym
import torch.nn as nn

from allenact.base_abstractions.preprocessor import Preprocessor
from allenact.embodiedai.sensors.vision_sensors import RGBSensor, DepthSensor
from allenact.utils.experiment_utils import Builder
from allenact_plugins.ithor_plugin.ithor_sensors import GoalObjectTypeThorSensor
from allenact_plugins.robothor_plugin.robothor_tasks import ObjectNavTask
from projects.objectnav_baselines.experiments.objectnav_base import ObjectNavBaseConfig
from allenact_plugins.navigation_plugin.objectnav.models import ObjectNavActorCritic


class ObjectNavMixInUnfrozenResNetGRUConfig(ObjectNavBaseConfig):
    """No ResNet preprocessor, using Raw Image as input, and learn a ResNet as
    encoder."""

    @classmethod
    def preprocessors(cls) -> Sequence[Union[Preprocessor, Builder[Preprocessor]]]:
        return []

    BACKBONE = (
        "gnresnet18"
        # "simple_cnn"
    )

    @classmethod
    def create_model(cls, **kwargs) -> nn.Module:
        rgb_uuid = next((s.uuid for s in cls.SENSORS if isinstance(s, RGBSensor)), None)
        depth_uuid = next(
            (s.uuid for s in cls.SENSORS if isinstance(s, DepthSensor)), None
        )
        goal_sensor_uuid = next(
            (s.uuid for s in cls.SENSORS if isinstance(s, GoalObjectTypeThorSensor)),
            None,
        )

        return ObjectNavActorCritic(
            action_space=gym.spaces.Discrete(len(ObjectNavTask.class_action_names())),
            observation_space=kwargs["sensor_preprocessor_graph"].observation_spaces,
            rgb_uuid=rgb_uuid,
            depth_uuid=depth_uuid,
            goal_sensor_uuid=goal_sensor_uuid,
            hidden_size=192
            if cls.MULTIPLE_BELIEFS and len(cls.AUXILIARY_UUIDS) > 1
            else 512,
            backbone=cls.BACKBONE,
            resnet_baseplanes=32,
            object_type_embedding_dim=32,
            num_rnn_layers=1,
            rnn_type="GRU",
            add_prev_actions=cls.ADD_PREV_ACTIONS,
            action_embed_size=6,
            auxiliary_uuids=cls.AUXILIARY_UUIDS,
            multiple_beliefs=cls.MULTIPLE_BELIEFS,
            beliefs_fusion=cls.BELIEF_FUSION,
        )
