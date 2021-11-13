from typing import Sequence, Union

import gym
import numpy as np
import torch.nn as nn

from allenact.base_abstractions.preprocessor import Preprocessor
from allenact.embodiedai.preprocessors.resnet import ClipResNetPreprocessor
from allenact.embodiedai.preprocessors.text_encoders import ClipTextPreprocessor
from allenact.embodiedai.sensors.vision_sensors import RGBSensor
from allenact.utils.experiment_utils import Builder
from allenact_plugins.ithor_plugin.ithor_sensors import GoalObjectTypeThorSensor
from allenact_plugins.robothor_plugin.robothor_tasks import ObjectNavTask
from projects.objectnav_baselines.experiments.objectnav_base import ObjectNavBaseConfig
from projects.objectnav_baselines.models.clip_models import CLIPObjectNavActorCritic


class ObjectNavZeroShotMixInClipGRUConfig(ObjectNavBaseConfig):
    CLIP_MODEL_TYPE: str

    @classmethod
    def preprocessors(cls) -> Sequence[Union[Preprocessor, Builder[Preprocessor]]]:
        preprocessors = []

        rgb_sensor = next((s for s in cls.SENSORS if isinstance(s, RGBSensor)), None)
        assert (
            np.linalg.norm(
                np.array(rgb_sensor._norm_means)
                - np.array(ClipResNetPreprocessor.CLIP_RGB_MEANS)
            )
            < 1e-5
        )
        assert (
            np.linalg.norm(
                np.array(rgb_sensor._norm_sds)
                - np.array(ClipResNetPreprocessor.CLIP_RGB_STDS)
            )
            < 1e-5
        )

        if rgb_sensor is not None:
            preprocessors.append(
                ClipResNetPreprocessor(
                    rgb_input_uuid=rgb_sensor.uuid,
                    clip_model_type=cls.CLIP_MODEL_TYPE,
                    pool=False,
                    output_uuid="rgb_clip_resnet",
                )
            )

        goal_sensor_uuid = next(
            (s.uuid for s in cls.SENSORS if isinstance(s, GoalObjectTypeThorSensor)),
            None,
        )
        preprocessors.append(
            ClipTextPreprocessor(
                goal_sensor_uuid=goal_sensor_uuid,
                object_types=cls.TARGET_TYPES,
                output_uuid="goal_object_encoded",
            )
        )

        return preprocessors

    @classmethod
    def create_model(cls, **kwargs) -> nn.Module:
        has_rgb = any(isinstance(s, RGBSensor) for s in cls.SENSORS)

        goal_sensor_uuid = next(
            (s.uuid for s in cls.SENSORS if isinstance(s, GoalObjectTypeThorSensor)),
            None,
        )
        goal_uuid = (
            'goal_object_encoded' if 'goal_object_encoded' in kwargs["sensor_preprocessor_graph"].preprocessors
            else goal_sensor_uuid
        )

        return CLIPObjectNavActorCritic(
            action_space=gym.spaces.Discrete(len(ObjectNavTask.class_action_names())),
            observation_space=kwargs["sensor_preprocessor_graph"].observation_spaces,
            goal_sensor_uuid=goal_uuid,
            rgb_resnet_preprocessor_uuid="rgb_clip_resnet" if has_rgb else None,
            hidden_size=512,
            goal_dims=32,
        )
