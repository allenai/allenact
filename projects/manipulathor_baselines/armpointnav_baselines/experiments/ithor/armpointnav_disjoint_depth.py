import gym
import torch.nn as nn

from allenact_plugins.manipulathor_plugin.manipulathor_constants import ENV_ARGS
from allenact_plugins.manipulathor_plugin.manipulathor_task_samplers import (
    ArmPointNavTaskSampler,
)
from projects.manipulathor_baselines.armpointnav_baselines.experiments.ithor.armpointnav_depth import (
    ArmPointNavDepth,
)
from projects.manipulathor_baselines.armpointnav_baselines.models.disjoint_arm_pointnav_models import (
    DisjointArmPointNavBaselineActorCritic,
)


class ArmPointNavDisjointDepth(ArmPointNavDepth):
    """An Object Navigation experiment configuration in iThor with RGB
    input."""

    TASK_SAMPLER = ArmPointNavTaskSampler

    def __init__(self):
        super().__init__()

        assert (
            self.CAMERA_WIDTH == 224
            and self.CAMERA_HEIGHT == 224
            and self.VISIBILITY_DISTANCE == 1
            and self.STEP_SIZE == 0.25
        )
        self.ENV_ARGS = {**ENV_ARGS, "renderDepthImage": True}

    @classmethod
    def create_model(cls, **kwargs) -> nn.Module:
        return DisjointArmPointNavBaselineActorCritic(
            action_space=gym.spaces.Discrete(
                len(cls.TASK_SAMPLER._TASK_TYPE.class_action_names())
            ),
            observation_space=kwargs["sensor_preprocessor_graph"].observation_spaces,
            hidden_size=512,
        )

    @classmethod
    def tag(cls):
        return cls.__name__
