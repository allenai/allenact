from typing import Sequence, Union, Type

import attr
import gym
import numpy as np
import torch.nn as nn

from allenact.base_abstractions.preprocessor import Preprocessor
from allenact.base_abstractions.sensor import Sensor
from allenact.embodiedai.sensors.vision_sensors import RGBSensor, DepthSensor
from allenact.utils.experiment_utils import Builder
from allenact_plugins.clip_plugin.clip_preprocessors import ClipResNetPreprocessor
from allenact_plugins.navigation_plugin.objectnav.models import (
    ResnetTensorNavActorCritic,
)


@attr.s(kw_only=True)
class ClipResNetPreprocessGRUActorCriticMixin:
    sensors: Sequence[Sensor] = attr.ib()
    clip_model_type: str = attr.ib()
    screen_size: int = attr.ib()
    goal_sensor_type: Type[Sensor] = attr.ib()
    pool: bool = attr.ib(default=False)

    def preprocessors(self) -> Sequence[Union[Preprocessor, Builder[Preprocessor]]]:
        preprocessors = []

        rgb_sensor = next((s for s in self.sensors if isinstance(s, RGBSensor)), None)
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
                    clip_model_type=self.clip_model_type,
                    pool=self.pool,
                    output_uuid="rgb_clip_resnet",
                )
            )

        depth_sensor = next(
            (s for s in self.sensors if isinstance(s, DepthSensor)), None
        )
        if depth_sensor is not None:
            preprocessors.append(
                ClipResNetPreprocessor(
                    rgb_input_uuid=depth_sensor.uuid,
                    clip_model_type=self.clip_model_type,
                    pool=self.pool,
                    output_uuid="depth_clip_resnet",
                )
            )

        return preprocessors

    def create_model(self, num_actions: int, add_prev_actions: bool, **kwargs) -> nn.Module:
        has_rgb = any(isinstance(s, RGBSensor) for s in self.sensors)
        has_depth = any(isinstance(s, DepthSensor) for s in self.sensors)

        goal_sensor_uuid = next(
            (s.uuid for s in self.sensors if isinstance(s, self.goal_sensor_type)),
            None,
        )

        return ResnetTensorNavActorCritic(
            action_space=gym.spaces.Discrete(num_actions),
            observation_space=kwargs["sensor_preprocessor_graph"].observation_spaces,
            goal_sensor_uuid=goal_sensor_uuid,
            rgb_resnet_preprocessor_uuid="rgb_clip_resnet" if has_rgb else None,
            depth_resnet_preprocessor_uuid="depth_clip_resnet" if has_depth else None,
            hidden_size=512,
            goal_dims=32,
            add_prev_actions=add_prev_actions
        )
