from allenact.utils.experiment_utils import TrainingPipeline
from allenact_plugins.habitat_plugin.habitat_sensors import RGBSensorHabitat
from allenact_plugins.habitat_plugin.habitat_sensors import (
    TargetCoordinatesSensorHabitat,
)
from projects.pointnav_baselines.experiments.habitat.pointnav_habitat_base import (
    PointNavHabitatBaseConfig,
)
from projects.pointnav_baselines.mixins import PointNavPPOMixin
from projects.pointnav_baselines.mixins import (
    PointNavUnfrozenResNetWithGRUActorCriticMixin,
)


class PointNavHabitatDepthDeterministiSimpleConvGRUDDPPOExperimentConfig(
    PointNavHabitatBaseConfig
):
    """An Point Navigation experiment configuration in Habitat with Depth
    input."""

    SENSORS = [
        RGBSensorHabitat(
            height=PointNavHabitatBaseConfig.SCREEN_SIZE,
            width=PointNavHabitatBaseConfig.SCREEN_SIZE,
            use_resnet_normalization=True,
        ),
        TargetCoordinatesSensorHabitat(coordinate_dims=2),
    ]

    def __init__(self):
        super().__init__()

        self.model_creation_handler = PointNavUnfrozenResNetWithGRUActorCriticMixin(
            backbone="simpleconv",
            sensors=self.SENSORS,
            auxiliary_uuids=[],
            add_prev_actions=True,
            multiple_beliefs=False,
            belief_fusion=None,
        )

    def training_pipeline(self, **kwargs) -> TrainingPipeline:
        return PointNavPPOMixin.training_pipeline(
            auxiliary_uuids=[],
            multiple_beliefs=False,
            normalize_advantage=True,
            advance_scene_rollout_period=self.ADVANCE_SCENE_ROLLOUT_PERIOD,
        )

    def create_model(self, **kwargs):
        return self.model_creation_handler.create_model(**kwargs)

    def tag(cls):
        return "PointNav-Habitat-RGB-SimpleConv-DDPPO"
