"""Experiment Config for MiniGrid tutorial."""

import gym
from torch import nn

from plugins.minigrid_plugin.minigrid_models import MiniGridSimpleConv
from projects.tutorials.minigrid_tutorial import MiniGridTutorialExperimentConfig
from plugins.minigrid_plugin.minigrid_tasks import MiniGridTask
from core.base_abstractions.sensor import SensorSuite


class MiniGridNoMemoryExperimentConfig(MiniGridTutorialExperimentConfig):
    @classmethod
    def tag(cls) -> str:
        return "MiniGridNoMemory"

    @classmethod
    def create_model(cls, **kwargs) -> nn.Module:
        return MiniGridSimpleConv(
            action_space=gym.spaces.Discrete(len(MiniGridTask.class_action_names())),
            observation_space=SensorSuite(cls.SENSORS).observation_spaces,
            num_objects=cls.SENSORS[0].num_objects,
            num_colors=cls.SENSORS[0].num_colors,
            num_states=cls.SENSORS[0].num_states,
        )
