import torch.nn as nn

from allenact.utils.experiment_utils import TrainingPipeline
from allenact_plugins.ithor_plugin.ithor_sensors import (
    RGBSensorThor,
    GoalObjectTypeThorSensor,
)
from projects.objectnav_baselines.experiments.robothor.objectnav_robothor_base import (
    ObjectNavRoboThorBaseConfig,
)
from projects.objectnav_baselines.mixins import (
    ObjectNavUnfrozenResNetWithGRUActorCriticMixin,
    ObjectNavPPOMixin,
)


class ObjectNavRoboThorRGBPPOExperimentConfig(ObjectNavRoboThorBaseConfig,):
    """An Object Navigation experiment configuration in RoboThor with RGB input
    without preprocessing by frozen ResNet (instead, a trainable ResNet)."""

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
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.model_creation_handler = ObjectNavUnfrozenResNetWithGRUActorCriticMixin(
            backbone="gnresnet18",
            sensors=self.SENSORS,
            auxiliary_uuids=[],
            add_prev_actions=True,
            multiple_beliefs=False,
            belief_fusion=None,
        )

    def training_pipeline(self, **kwargs) -> TrainingPipeline:
        return ObjectNavPPOMixin.training_pipeline(
            auxiliary_uuids=[],
            multiple_beliefs=False,
            advance_scene_rollout_period=self.ADVANCE_SCENE_ROLLOUT_PERIOD,
        )

    def create_model(self, **kwargs) -> nn.Module:
        return self.model_creation_handler.create_model(**kwargs)

    def tag(self):
        return "ObjectNav-RoboTHOR-RGB-UnfrozenResNet18GRU-DDPPO"
