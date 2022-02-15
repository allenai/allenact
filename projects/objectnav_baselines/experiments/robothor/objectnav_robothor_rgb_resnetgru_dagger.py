from allenact.base_abstractions.sensor import ExpertActionSensor
from allenact_plugins.ithor_plugin.ithor_sensors import (
    RGBSensorThor,
    GoalObjectTypeThorSensor,
)
from allenact_plugins.robothor_plugin.robothor_tasks import ObjectNavTask
from projects.objectnav_baselines.experiments.objectnav_mixin_dagger import (
    ObjectNavMixInDAggerConfig,
)
from projects.objectnav_baselines.experiments.objectnav_mixin_resnet18gru import (
    ObjectNavMixInResNet18GRUConfig,
)
from projects.objectnav_baselines.experiments.robothor.objectnav_robothor_base import (
    ObjectNavRoboThorBaseConfig,
)


class ObjectNaviThorRGBDAggerExperimentConfig(
    ObjectNavRoboThorBaseConfig,
    ObjectNavMixInDAggerConfig,
    ObjectNavMixInResNet18GRUConfig,
):
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
        ExpertActionSensor(nactions=len(ObjectNavTask.class_action_names()),),
    ]

    @classmethod
    def tag(cls):
        return "Objectnav-RoboTHOR-RGB-ResNetGRU-DAgger"
