from allenact.utils.experiment_utils import TrainingPipeline
from allenact_plugins.habitat_plugin.habitat_sensors import (
    DepthSensorHabitat,
    TargetCoordinatesSensorHabitat,
)
from projects.pointnav_baselines.experiments.habitat.pointnav_habitat_base import (
    PointNavHabitatBaseConfig,
)
from projects.pointnav_baselines.mixins import (
    PointNavPPOMixin,
    PointNavUnfrozenResNetWithGRUActorCriticMixin,
)


class PointNavHabitatDepthDeterministiSimpleConvGRUDDPPOExperimentConfig(
    PointNavHabitatBaseConfig,
):
    """An Point Navigation experiment configuration in Habitat with Depth
    input."""

    SENSORS = [
        DepthSensorHabitat(
            height=PointNavHabitatBaseConfig.SCREEN_SIZE,
            width=PointNavHabitatBaseConfig.SCREEN_SIZE,
            use_normalization=True,
        ),
        TargetCoordinatesSensorHabitat(coordinate_dims=2),
    ]

    def __init__(self):
        super().__init__()

        self.model_creation_handler = PointNavUnfrozenResNetWithGRUActorCriticMixin(
            backbone="simple_cnn",
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

    def tag(self):
        return "PointNav-Habitat-Depth-SimpleConv-DDPPO"
