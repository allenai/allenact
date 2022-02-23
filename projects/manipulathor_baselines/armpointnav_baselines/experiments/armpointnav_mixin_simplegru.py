from typing import Sequence, Union

import gym
import torch.nn as nn

from allenact.base_abstractions.preprocessor import Preprocessor
from allenact.base_abstractions.task import TaskSampler
from allenact.utils.experiment_utils import Builder
from projects.manipulathor_baselines.armpointnav_baselines.experiments.armpointnav_base import (
    ArmPointNavBaseConfig,
)
from projects.manipulathor_baselines.armpointnav_baselines.models.arm_pointnav_models import (
    ArmPointNavBaselineActorCritic,
)


class ArmPointNavMixInSimpleGRUConfig(ArmPointNavBaseConfig):
    TASK_SAMPLER: TaskSampler

    @classmethod
    def preprocessors(cls) -> Sequence[Union[Preprocessor, Builder[Preprocessor]]]:
        preprocessors = []
        return preprocessors

    @classmethod
    def create_model(cls, **kwargs) -> nn.Module:

        return ArmPointNavBaselineActorCritic(
            action_space=gym.spaces.Discrete(
                len(cls.TASK_SAMPLER._TASK_TYPE.class_action_names())
            ),
            observation_space=kwargs["sensor_preprocessor_graph"].observation_spaces,
            hidden_size=512,
        )
