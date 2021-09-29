from typing import Sequence, Union

import gym
import numpy as np
import torch.nn as nn

from allenact.base_abstractions.preprocessor import Preprocessor
from allenact.embodiedai.preprocessors.resnet import ClipResNetPreprocessor
from allenact.embodiedai.sensors.vision_sensors import RGBSensor, DepthSensor
from allenact.utils.experiment_utils import Builder
from allenact_plugins.ithor_plugin.ithor_sensors import GoalObjectTypeThorSensor
from allenact_plugins.robothor_plugin.robothor_tasks import ObjectNavTask
from projects.objectnav_baselines.experiments.objectnav_base import ObjectNavBaseConfig
from projects.objectnav_baselines.experiments.objectnav_mixin_clipresnet50gru import (
    CLIP_RGB_MEANS,
    CLIP_RGB_STDS,
)
from projects.objectnav_baselines.models.object_nav_models import (
    ResnetTensorObjectNavActorCritic,
)


class ObjectNavMixInClipResNet50x16GRUConfig(ObjectNavBaseConfig):
    @classmethod
    def preprocessors(cls) -> Sequence[Union[Preprocessor, Builder[Preprocessor]]]:
        preprocessors = []

        rgb_sensor = next((s for s in cls.SENSORS if isinstance(s, RGBSensor)), None)
        assert (
            np.linalg.norm(np.array(rgb_sensor._norm_means) - np.array(CLIP_RGB_MEANS))
            < 1e-5
        )
        assert (
            np.linalg.norm(np.array(rgb_sensor._norm_sds) - np.array(CLIP_RGB_STDS))
            < 1e-5
        )

        if rgb_sensor is not None:
            preprocessors.append(
                ClipResNetPreprocessor(
                    rgb_input_uuid=rgb_sensor.uuid,
                    clip_model_type="RN50x16",
                    pool=False,
                    output_uuid="rgb_clip_resnet",
                    output_shape=(3072, 7, 7)
                )
            )

        depth_sensor = next(
            (s for s in cls.SENSORS if isinstance(s, DepthSensor)), None
        )
        assert depth_sensor is None

        return preprocessors

    @classmethod
    def create_model(cls, **kwargs) -> nn.Module:
        has_rgb = any(isinstance(s, RGBSensor) for s in cls.SENSORS)
        has_depth = any(isinstance(s, DepthSensor) for s in cls.SENSORS)
        assert not has_depth

        goal_sensor_uuid = next(
            (s.uuid for s in cls.SENSORS if isinstance(s, GoalObjectTypeThorSensor)),
            None,
        )

        return ResnetTensorObjectNavActorCritic(
            action_space=gym.spaces.Discrete(len(ObjectNavTask.class_action_names())),
            observation_space=kwargs["sensor_preprocessor_graph"].observation_spaces,
            goal_sensor_uuid=goal_sensor_uuid,
            rgb_resnet_preprocessor_uuid="rgb_clip_resnet" if has_rgb else None,
            depth_resnet_preprocessor_uuid=None,
            hidden_size=512,
            goal_dims=32,
        )
