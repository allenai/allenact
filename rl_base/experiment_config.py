import abc
import torch.nn as nn
from typing import Dict, Any


class ExperimentConfig(abc.ABC):
    @classmethod
    def training_pipeline(cls, **kwargs):
        raise NotImplementedError()

    @classmethod
    def create_model(cls, **kwargs) -> nn.Module:
        raise NotImplementedError()

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
