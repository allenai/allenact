from allenact.utils.experiment_utils import TrainingPipeline
from allenact_plugins.robothor_plugin.robothor_sensors import (
    DepthSensorThor,
    GPSCompassSensorRoboThor,
)
from projects.pointnav_baselines.mixins import (
    PointNavUnfrozenResNetWithGRUActorCriticMixin,
)
from projects.pointnav_baselines.experiments.ithor.pointnav_ithor_base import (
    PointNaviThorBaseConfig,
)
from projects.pointnav_baselines.mixins import PointNavPPOMixin


class PointNaviThorDepthPPOExperimentConfig(PointNaviThorBaseConfig):
    """An Point Navigation experiment configuration in iThor with Depth
    input."""

    SENSORS = [
        DepthSensorThor(
            height=PointNaviThorBaseConfig.SCREEN_SIZE,
            width=PointNaviThorBaseConfig.SCREEN_SIZE,
            use_normalization=True,
            uuid="depth_lowres",
        ),
        GPSCompassSensorRoboThor(),
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
        return "PointNav-iTHOR-Depth-SimpleConv-DDPPO"
