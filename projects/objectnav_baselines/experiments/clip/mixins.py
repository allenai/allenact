from typing import Sequence, Union, Type, Tuple, Optional, Dict, Any

import attr
import gym
import numpy as np
import torch
import torch.nn as nn

from allenact.base_abstractions.distributions import CategoricalDistr
from allenact.base_abstractions.misc import (
    ObservationType,
    Memory,
    ActorCriticOutput,
    DistributionType,
)
from allenact.base_abstractions.preprocessor import Preprocessor
from allenact.base_abstractions.sensor import Sensor
from allenact.embodiedai.sensors.vision_sensors import RGBSensor, DepthSensor
from allenact.utils.experiment_utils import Builder
from allenact.utils.misc_utils import prepare_locals_for_super
from allenact_plugins.clip_plugin.clip_preprocessors import ClipResNetPreprocessor
from allenact_plugins.navigation_plugin.objectnav.models import (
    ResnetTensorNavActorCritic,
)


class LookDownFirstResnetTensorNavActorCritic(ResnetTensorNavActorCritic):
    def __init__(self, look_down_action_index: int, **kwargs):
        super().__init__(**kwargs)

        self.look_down_action_index = look_down_action_index
        self.register_buffer(
            "look_down_delta", torch.zeros(1, 1, self.action_space.n), persistent=False
        )
        self.look_down_delta[0, 0, self.look_down_action_index] = 99999

    def forward(  # type:ignore
        self,
        observations: ObservationType,
        memory: Memory,
        prev_actions: torch.Tensor,
        masks: torch.FloatTensor,
    ) -> Tuple[ActorCriticOutput[DistributionType], Optional[Memory]]:
        ac_out, memory = super(LookDownFirstResnetTensorNavActorCritic, self).forward(
            **prepare_locals_for_super(locals())
        )

        logits = ac_out.distributions.logits * masks + self.look_down_delta * (
            1 - masks
        )
        ac_out = ActorCriticOutput(
            distributions=CategoricalDistr(logits=logits),
            values=ac_out.values,
            extras=ac_out.extras,
        )

        return ac_out, memory


@attr.s(kw_only=True)
class ClipResNetPreprocessGRUActorCriticMixin:
    sensors: Sequence[Sensor] = attr.ib()
    clip_model_type: str = attr.ib()
    screen_size: int = attr.ib()
    goal_sensor_type: Type[Optional[Sensor]] = attr.ib()
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
                    input_img_height_width=(rgb_sensor.height, rgb_sensor.width),
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
                    input_img_height_width=(depth_sensor.height, depth_sensor.width),
                )
            )

        return preprocessors

    def create_model(
        self,
        num_actions: int,
        add_prev_actions: bool,
        look_down_first: bool = False,
        look_down_action_index: Optional[int] = None,
        hidden_size: int = 512,
        rnn_type="GRU",
        model_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> nn.Module:
        has_rgb = any(isinstance(s, RGBSensor) for s in self.sensors)
        has_depth = any(isinstance(s, DepthSensor) for s in self.sensors)

        goal_sensor_uuid = next(
            (s.uuid for s in self.sensors if isinstance(s, self.goal_sensor_type)),
            None,
        )

        if model_kwargs is None:
            model_kwargs = {}

        model_kwargs = dict(
            action_space=gym.spaces.Discrete(num_actions),
            observation_space=kwargs["sensor_preprocessor_graph"].observation_spaces,
            goal_sensor_uuid=goal_sensor_uuid,
            rgb_resnet_preprocessor_uuid="rgb_clip_resnet" if has_rgb else None,
            depth_resnet_preprocessor_uuid="depth_clip_resnet" if has_depth else None,
            hidden_size=hidden_size,
            goal_dims=32,
            add_prev_actions=add_prev_actions,
            rnn_type=rnn_type,
            **model_kwargs
        )

        if not look_down_first:
            return ResnetTensorNavActorCritic(**model_kwargs)
        else:
            return LookDownFirstResnetTensorNavActorCritic(
                look_down_action_index=look_down_action_index, **model_kwargs
            )
