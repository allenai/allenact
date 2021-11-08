from typing import Sequence, Union

import gym
import torch.nn as nn
from torchvision import models

from allenact.base_abstractions.preprocessor import Preprocessor
from allenact.embodiedai.preprocessors.resnet import ResNetPreprocessor
from allenact.embodiedai.sensors.vision_sensors import RGBSensor, DepthSensor
from allenact.utils.experiment_utils import Builder
from allenact_plugins.ithor_plugin.ithor_sensors import GoalObjectTypeThorSensor
from allenact_plugins.robothor_plugin.robothor_tasks import ObjectNavTask
from projects.objectnav_baselines.experiments.objectnav_base import ObjectNavBaseConfig
from projects.objectnav_baselines.experiments.robothor.objectnav_robothor_base import (
    ObjectNavRoboThorBaseConfig,
)
from projects.objectnav_baselines.models.object_nav_models import (
    ResnetTensorObjectNavActorCritic,
)


class ObjectNavMixInResNetGRUConfig(ObjectNavBaseConfig):
    RESNET_TYPE: str

    @classmethod
    def preprocessors(cls) -> Sequence[Union[Preprocessor, Builder[Preprocessor]]]:
        if not hasattr(cls, "RESNET_TYPE"):
            raise RuntimeError(
                "When subclassing `ObjectNavMixInResNetGRUConfig` we now expect that you have specified `RESNET_TYPE`"
                " as a class variable of your subclass (e.g. `RESNET_TYPE = 'RN18'` to use a ResNet18 model)."
                " Alternatively you can instead subclass `ObjectNavMixInResNet18GRUConfig` which does this"
                " specification for you."
            )

        preprocessors = []

        if cls.RESNET_TYPE in ["RN18", "RN34"]:
            output_shape = (512, 7, 7)
        elif cls.RESNET_TYPE in ["RN50", "RN101", "RN152"]:
            output_shape = (2048, 7, 7)
        else:
            raise NotImplementedError(
                f"`RESNET_TYPE` must be one 'RNx' with x equaling one of"
                f" 18, 34, 50, 101, or 152."
            )

        rgb_sensor = next((s for s in cls.SENSORS if isinstance(s, RGBSensor)), None)
        if rgb_sensor is not None:
            preprocessors.append(
                ResNetPreprocessor(
                    input_height=cls.SCREEN_SIZE,
                    input_width=cls.SCREEN_SIZE,
                    output_width=output_shape[2],
                    output_height=output_shape[1],
                    output_dims=output_shape[0],
                    pool=False,
                    torchvision_resnet_model=getattr(
                        models, f"resnet{cls.RESNET_TYPE.replace('RN', '')}"
                    ),
                    input_uuids=[rgb_sensor.uuid],
                    output_uuid="rgb_resnet_imagenet",
                )
            )

        depth_sensor = next(
            (s for s in cls.SENSORS if isinstance(s, DepthSensor)), None
        )
        if depth_sensor is not None:
            preprocessors.append(
                ResNetPreprocessor(
                    input_height=cls.SCREEN_SIZE,
                    input_width=cls.SCREEN_SIZE,
                    output_width=output_shape[2],
                    output_height=output_shape[1],
                    output_dims=output_shape[0],
                    pool=False,
                    torchvision_resnet_model=getattr(
                        models, f"resnet{cls.RESNET_TYPE.replace('RN', '')}"
                    ),
                    input_uuids=[depth_sensor.uuid],
                    output_uuid="depth_resnet_imagenet",
                )
            )

        return preprocessors

    @classmethod
    def create_model(cls, **kwargs) -> nn.Module:
        has_rgb = any(isinstance(s, RGBSensor) for s in cls.SENSORS)
        has_depth = any(isinstance(s, DepthSensor) for s in cls.SENSORS)
        goal_sensor_uuid = next(
            (s.uuid for s in cls.SENSORS if isinstance(s, GoalObjectTypeThorSensor)),
            None,
        )

        return ResnetTensorObjectNavActorCritic(
            action_space=gym.spaces.Discrete(len(ObjectNavTask.class_action_names())),
            observation_space=kwargs["sensor_preprocessor_graph"].observation_spaces,
            goal_sensor_uuid=goal_sensor_uuid,
            rgb_resnet_preprocessor_uuid="rgb_resnet_imagenet" if has_rgb else None,
            depth_resnet_preprocessor_uuid="depth_resnet_imagenet"
            if has_depth
            else None,
            hidden_size=512,
            goal_dims=32,
        )
