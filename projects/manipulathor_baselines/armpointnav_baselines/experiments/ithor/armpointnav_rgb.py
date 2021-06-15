from allenact_plugins.ithor_plugin.ithor_sensors import RGBSensorThor
from allenact_plugins.manipulathor_plugin.manipulathor_constants import ENV_ARGS
from allenact_plugins.manipulathor_plugin.manipulathor_sensors import DepthSensorThor, RelativeAgentArmToObjectSensor, RelativeObjectToGoalSensor, PickedUpObjSensor
from allenact_plugins.manipulathor_plugin.manipulathor_task_samplers import ArmPointNavTaskSampler
from projects.manipulathor_baselines.armpointnav_baselines.experiments.armpointnav_mixin_ddppo import ArmPointNavMixInPPOConfig
from projects.manipulathor_baselines.armpointnav_baselines.experiments.armpointnav_mixin_simplegru import ArmPointNavMixInSimpleGRUConfig
from projects.manipulathor_baselines.armpointnav_baselines.experiments.ithor.armpointnav_ithor_base import ArmPointNaviThorBaseConfig


class ArmPointNavRGB(
    ArmPointNaviThorBaseConfig,
    ArmPointNavMixInPPOConfig,
    ArmPointNavMixInSimpleGRUConfig,
):
    """An Object Navigation experiment configuration in iThor with RGB
    input."""

    SENSORS = [
        RGBSensorThor(
            height=ArmPointNaviThorBaseConfig.SCREEN_SIZE,
            width=ArmPointNaviThorBaseConfig.SCREEN_SIZE,
            use_resnet_normalization=True,
            uuid="rgb_lowres",
        ),
        RelativeAgentArmToObjectSensor(),
        RelativeObjectToGoalSensor(),
        PickedUpObjSensor(),
    ]

    MAX_STEPS = 200
    TASK_SAMPLER = ArmPointNavTaskSampler  #

    def __init__(self):
        super().__init__()

        assert (
            self.CAMERA_WIDTH == 224
            and self.CAMERA_HEIGHT == 224
            and self.VISIBILITY_DISTANCE == 1
            and self.STEP_SIZE == 0.25
        )
        self.ENV_ARGS = {**ENV_ARGS}

    @classmethod
    def tag(cls):
        return cls.__name__
