from typing import Sequence, Union

import torch.nn as nn
from torch.distributions.utils import lazy_property

from allenact.base_abstractions.preprocessor import Preprocessor
from allenact.utils.experiment_utils import Builder, TrainingPipeline
from allenact_plugins.clip_plugin.clip_preprocessors import ClipResNetPreprocessor
from allenact_plugins.habitat_plugin.habitat_sensors import (
    RGBSensorHabitat,
    TargetObjectSensorHabitat,
)
from projects.objectnav_baselines.experiments.clip.mixins import (
    ClipResNetPreprocessGRUActorCriticMixin,
)
from projects.objectnav_baselines.experiments.habitat.objectnav_habitat_base import (
    ObjectNavHabitatBaseConfig,
)
from projects.objectnav_baselines.mixins import ObjectNavPPOMixin


class ObjectNavHabitatRGBClipResNet50GRUDDPPOExperimentConfig(
    ObjectNavHabitatBaseConfig
):
    """An Object Navigation experiment configuration in Habitat."""

    CLIP_MODEL_TYPE = "RN50"

    def __init__(self, lr: float, **kwargs):
        super().__init__(**kwargs)

        self.lr = lr

        self.preprocessing_and_model = ClipResNetPreprocessGRUActorCriticMixin(
            sensors=self.SENSORS,
            clip_model_type=self.CLIP_MODEL_TYPE,
            screen_size=self.SCREEN_SIZE,
            goal_sensor_type=TargetObjectSensorHabitat,
        )

    @lazy_property
    def SENSORS(self):
        return [
            RGBSensorHabitat(
                height=ObjectNavHabitatBaseConfig.SCREEN_SIZE,
                width=ObjectNavHabitatBaseConfig.SCREEN_SIZE,
                use_resnet_normalization=True,
                mean=ClipResNetPreprocessor.CLIP_RGB_MEANS,
                stdev=ClipResNetPreprocessor.CLIP_RGB_STDS,
            ),
            TargetObjectSensorHabitat(len(self.DEFAULT_OBJECT_CATEGORIES_TO_IND)),
        ]

    def training_pipeline(self, **kwargs) -> TrainingPipeline:
        return ObjectNavPPOMixin.training_pipeline(
            lr=self.lr,
            auxiliary_uuids=[],
            multiple_beliefs=False,
            advance_scene_rollout_period=self.ADVANCE_SCENE_ROLLOUT_PERIOD,
        )

    def preprocessors(self) -> Sequence[Union[Preprocessor, Builder[Preprocessor]]]:
        return self.preprocessing_and_model.preprocessors()

    def create_model(self, **kwargs) -> nn.Module:
        return self.preprocessing_and_model.create_model(
            num_actions=self.ACTION_SPACE.n, add_prev_actions=self.add_prev_actions, **kwargs
        )

    def tag(self):
        return (
            f"{super(ObjectNavHabitatRGBClipResNet50GRUDDPPOExperimentConfig, self).tag()}"
            f"-RGB-ClipResNet50GRU-DDPPO-lr{self.lr}"
        )
