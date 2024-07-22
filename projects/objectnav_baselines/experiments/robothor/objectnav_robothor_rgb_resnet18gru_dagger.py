from typing import Sequence, Union

import torch.nn as nn

from allenact.base_abstractions.preprocessor import Preprocessor
from allenact.base_abstractions.sensor import ExpertActionSensor
from allenact.utils.experiment_utils import Builder, TrainingPipeline
from allenact_plugins.ithor_plugin.ithor_sensors import (
    GoalObjectTypeThorSensor,
    RGBSensorThor,
)
from allenact_plugins.robothor_plugin.robothor_tasks import ObjectNavTask
from projects.objectnav_baselines.experiments.robothor.objectnav_robothor_base import (
    ObjectNavRoboThorBaseConfig,
)
from projects.objectnav_baselines.mixins import (
    ResNetPreprocessGRUActorCriticMixin,
    ObjectNavDAggerMixin,
)


class ObjectNavRoboThorRGBDAggerExperimentConfig(ObjectNavRoboThorBaseConfig):
    """An Object Navigation experiment configuration in RoboThor with RGB
    input."""

    SENSORS = [
        RGBSensorThor(
            height=ObjectNavRoboThorBaseConfig.SCREEN_SIZE,
            width=ObjectNavRoboThorBaseConfig.SCREEN_SIZE,
            use_resnet_normalization=True,
            uuid="rgb_lowres",
        ),
        GoalObjectTypeThorSensor(
            object_types=ObjectNavRoboThorBaseConfig.TARGET_TYPES,
        ),
        ExpertActionSensor(
            nactions=len(ObjectNavTask.class_action_names()),
        ),
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.REWARD_CONFIG["shaping"] = 0

        self.preprocessing_and_model = ResNetPreprocessGRUActorCriticMixin(
            sensors=self.SENSORS,
            resnet_type="RN18",
            screen_size=self.SCREEN_SIZE,
            goal_sensor_type=GoalObjectTypeThorSensor,
        )

    def training_pipeline(self, **kwargs) -> TrainingPipeline:
        return ObjectNavDAggerMixin.training_pipeline(
            advance_scene_rollout_period=self.ADVANCE_SCENE_ROLLOUT_PERIOD,
        )

    def preprocessors(self) -> Sequence[Union[Preprocessor, Builder[Preprocessor]]]:
        return self.preprocessing_and_model.preprocessors()

    def create_model(self, **kwargs) -> nn.Module:
        return self.preprocessing_and_model.create_model(
            num_actions=self.ACTION_SPACE.n, **kwargs
        )

    @classmethod
    def tag(cls):
        return "ObjectNav-RoboTHOR-RGB-ResNet18GRU-DAgger"
