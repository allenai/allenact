import abc
from abc import ABC
from typing import Dict, Sequence, Optional, List, Any, cast

import gym
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

from allenact.algorithms.onpolicy_sync.losses.ppo import PPO

from allenact.algorithms.onpolicy_sync.losses.a2cacktr import A2C
from allenact.algorithms.onpolicy_sync.losses.a2cacktr import A2CConfig

from allenact.base_abstractions.experiment_config import ExperimentConfig, TaskSampler
from allenact.base_abstractions.sensor import Sensor, SensorSuite
from allenact_plugins.gym_plugin.gym_models import MemorylessActorCritic
from allenact_plugins.gym_plugin.gym_sensors import GymMuJoCoSensor

from allenact_plugins.gym_plugin.gym_tasks import GymTaskSampler
from allenact.utils.experiment_utils import (
    TrainingPipeline,
    Builder,
    PipelineStage,
    LinearDecay,
)
from allenact.utils.viz_utils import VizSuite, AgentViewViz


class GymBaseConfig(ExperimentConfig):

    SENSORS: Optional[Sequence[Sensor]] = None

    # @abc.abstractmethod
    def _get_sampler_args(
        self, process_ind: int, mode: str, seeds: List[int]
    ) -> Dict[str, Any]:
        raise NotImplementedError

    def train_task_sampler_args(
        self,
        process_ind: int,
        total_processes: int,
        devices: Optional[List[int]] = None,
        seeds: Optional[List[int]] = None,
        deterministic_cudnn: bool = False,
    ) -> Dict[str, Any]:
        return self._get_sampler_args(
            process_ind=process_ind, mode="train", seeds=seeds
        )

    def valid_task_sampler_args(
        self,
        process_ind: int,
        total_processes: int,
        devices: Optional[List[int]] = None,
        seeds: Optional[List[int]] = None,
        deterministic_cudnn: bool = False,
    ) -> Dict[str, Any]:
        return self._get_sampler_args(
            process_ind=process_ind, mode="valid", seeds=seeds
        )

    def test_task_sampler_args(
        self,
        process_ind: int,
        total_processes: int,
        devices: Optional[List[int]] = None,
        seeds: Optional[List[int]] = None,
        deterministic_cudnn: bool = False,
    ) -> Dict[str, Any]:
        return self._get_sampler_args(process_ind=process_ind, mode="test", seeds=seeds)
