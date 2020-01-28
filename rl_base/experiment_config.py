import abc
from typing import Dict, Any, Callable
from rl_base.task import TaskSampler

import torch.nn as nn


class ExperimentConfig(abc.ABC):
    @classmethod
    @abc.abstractmethod
    def tag(cls):
        raise NotImplementedError()

    @classmethod
    @abc.abstractmethod
    def training_pipeline(cls, **kwargs):
        raise NotImplementedError()

    @classmethod
    def create_model(cls, **kwargs) -> nn.Module:
        raise NotImplementedError()

    @staticmethod
    def make_sampler_fn(**kwargs) -> TaskSampler:
        raise NotImplementedError

    def train_task_sampler_args(
        self, process_ind: int, total_processes: int
    ) -> Dict[str, Any]:
        raise NotImplementedError()

    def valid_task_sampler_args(
        self, process_ind: int, total_processes: int
    ) -> Dict[str, Any]:
        raise NotImplementedError()

    def test_task_sampler_args(
        self, process_ind: int, total_processes: int
    ) -> Dict[str, Any]:
        raise NotImplementedError()
