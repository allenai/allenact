"""Defines the `ExperimentConfig` abstract class used as the basis of all
experiments."""

import abc
from typing import Dict, Any, Optional, List
from rl_base.task import TaskSampler

import torch.nn as nn


class ExperimentConfig(abc.ABC):
    """Abstract class used to define experiments.

    Instead of using yaml or text files, experiments in our framework
    are defined as a class. In particular, to define an experiment one
    must define a new class inheriting from this class which implements
    all of the below methods. The below methods will then be called when
    running the experiment.
    """

    @classmethod
    @abc.abstractmethod
    def tag(cls) -> str:
        """A string describing the experiment."""
        raise NotImplementedError()

    @classmethod
    @abc.abstractmethod
    def training_pipeline(cls, **kwargs):
        """

        @param kwargs:
        @return:
        """
        raise NotImplementedError()

    @classmethod
    @abc.abstractmethod
    def machine_params(cls, mode="train", **kwargs):
        raise NotImplementedError()

    @classmethod
    @abc.abstractmethod
    def create_model(cls, **kwargs) -> nn.Module:
        raise NotImplementedError()

    @staticmethod
    def make_sampler_fn(**kwargs) -> TaskSampler:
        raise NotImplementedError

    def train_task_sampler_args(
        self,
        process_ind: int,
        total_processes: int,
        devices: Optional[List[int]],
        seeds: Optional[List[int]],
        deterministic_cudnn: bool,
    ) -> Dict[str, Any]:
        raise NotImplementedError()

    def valid_task_sampler_args(
        self,
        process_ind: int,
        total_processes: int,
        devices: Optional[List[int]],
        seeds: Optional[List[int]],
        deterministic_cudnn: bool,
    ) -> Dict[str, Any]:
        raise NotImplementedError()

    def test_task_sampler_args(
        self,
        process_ind: int,
        total_processes: int,
        devices: Optional[List[int]],
        seeds: Optional[List[int]],
        deterministic_cudnn: bool,
    ) -> Dict[str, Any]:
        raise NotImplementedError()
