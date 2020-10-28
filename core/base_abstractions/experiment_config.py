"""Defines the `ExperimentConfig` abstract class used as the basis of all
experiments."""

import abc
from typing import Dict, Any, Optional, List, Union, Sequence, Tuple, cast

import torch
import torch.nn as nn

from core.base_abstractions.preprocessor import ObservationSet
from core.base_abstractions.task import TaskSampler
from utils.experiment_utils import TrainingPipeline, Builder
from utils.system import get_logger
from utils.viz_utils import VizSuite


class MachineParams(object):
    def __init__(
        self,
        nprocesses: Union[int, Sequence[int]],
        devices: Union[
            None, int, str, torch.device, Sequence[Union[int, str, torch.device]]
        ] = None,
        observation_set: Optional[
            Union[ObservationSet, Builder[ObservationSet]]
        ] = None,
        sampler_devices: Union[
            None, int, str, torch.device, Sequence[Union[int, str, torch.device]]
        ] = None,
        visualizer: Optional[Union[VizSuite, Builder[VizSuite]]] = None,
        gpu_ids: Union[int, Sequence[int]] = None,
    ):
        assert (
            gpu_ids is None or devices is None
        ), "only one of `gpu_ids` or `devices` should be set."
        if gpu_ids is not None:
            get_logger().warning(
                "The `gpu_ids` parameter will be deprecated, use `devices` instead."
            )
            devices = gpu_ids

        self.nprocesses = (
            nprocesses if isinstance(nprocesses, Sequence) else (nprocesses,)
        )

        self.devices: Tuple[torch.device, ...] = self._standardize_devices(
            devices=devices, nworkers=len(self.nprocesses)
        )

        self._observation_set_maybe_builder = observation_set
        self.sampler_devices: Tuple[
            torch.device, ...
        ] = None if sampler_devices is None else self._standardize_devices(
            devices=sampler_devices, nworkers=len(self.nprocesses)
        )
        self._visualizer_maybe_builder = visualizer

        self._observation_set_cached: Optional[ObservationSet] = None
        self._visualizer_cached: Optional[VizSuite] = None

    @classmethod
    def instance_from(
        cls, machine_params: Union["MachineParams", Dict[str, Any]]
    ) -> "MachineParams":
        if isinstance(machine_params, cls):
            return machine_params
        assert isinstance(machine_params, Dict)
        return cls(**machine_params)

    @staticmethod
    def _standardize_devices(
        devices: Optional[
            Union[int, str, torch.device, Sequence[Union[int, str, torch.device]]]
        ],
        nworkers: int,
    ) -> Tuple[torch.device, ...]:
        if devices is None or (isinstance(devices, Sequence) and len(devices) == 0):
            devices = torch.device("cpu")

        if not isinstance(devices, Sequence):
            devices = (devices,) * nworkers

        assert (
            len(devices) == nworkers
        ), f"The number of devices ({len(devices)}) must equal the number of workers ({nworkers})"

        devices = tuple(
            torch.device("cpu") if d == -1 else torch.device(d) for d in devices  # type: ignore
        )
        for d in devices:
            if d != torch.device("cpu"):
                try:
                    torch.cuda.get_device_capability(d)  # type: ignore
                except Exception:
                    raise RuntimeError(
                        f"It appears the cuda device {d} is not available on your system."
                    )

        return cast(Tuple[torch.device, ...], devices)

    @property
    def observation_set(self) -> Optional[ObservationSet]:
        if self._observation_set_maybe_builder is None:
            return None

        if self._observation_set_cached is None:
            if isinstance(self._observation_set_maybe_builder, Builder):
                self._observation_set_cached = self._observation_set_maybe_builder()
            else:
                self._observation_set_cached = self._observation_set_maybe_builder

        return self._observation_set_cached

    @property
    def visualizer(self) -> Optional[VizSuite]:
        if self._visualizer_maybe_builder is None:
            return None

        if self._visualizer_cached is None:
            if isinstance(self._visualizer_maybe_builder, Builder):
                self._visualizer_cached = self._visualizer_maybe_builder()
            else:
                self._visualizer_cached = self._visualizer_maybe_builder

        return self._visualizer_cached


class FrozenClassVariables(abc.ABCMeta):
    """Metaclass for ExperimentConfig.

    Ensures ExperimentConfig class-level attributes cannot be modified.
    ExperimentConfig attributes can still be modified at the object
    level.
    """

    def __setattr__(cls, attr, value):
        if isinstance(cls, type) and (
            attr != "__abstractmethods__" and not attr.startswith("_abc_")
        ):
            raise RuntimeError(
                "Cannot edit class-level attributes.\n"
                "Changing the values of class-level attributes is disabled in ExperimentConfig classes.\n"
                "This is to prevent problems that can occur otherwise when using multiprocessing.\n"
                "If you wish to change the value of a configuration, please do so for an instance of that"
                "configuration.\nTriggered by attempting to modify {}".format(
                    cls.__name__
                )
            )
        else:
            super().__setattr__(attr, value)


class ExperimentConfig(metaclass=FrozenClassVariables):
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
    def training_pipeline(cls, **kwargs) -> TrainingPipeline:
        """Creates the training pipeline.

        # Parameters

        kwargs : Extra kwargs. Currently unused.

        # Returns

        An instantiate `TrainingPipeline` object.
        """
        raise NotImplementedError()

    @classmethod
    @abc.abstractmethod
    def machine_params(
        cls, mode="train", **kwargs
    ) -> Union[MachineParams, Dict[str, Any]]:
        """Parameters used to specify machine information.

        Machine information includes at least (1) the number of processes
        to train with and (2) the gpu devices indices to use.

        mode : Whether or not the machine parameters should be those for
            "train", "valid", or "test".
        kwargs : Extra kwargs.

        # Returns

        A dictionary of the form `{"nprocesses": ..., "gpu_ids": ..., ...}`.
        Here `nprocesses` must be a non-negative integer, `gpu_ids` must
        be a sequence of non-negative integers (if empty, then everything
        will be run on the cpu).
        """
        raise NotImplementedError()

    @classmethod
    @abc.abstractmethod
    def create_model(cls, **kwargs) -> nn.Module:
        """Create the neural model."""
        raise NotImplementedError()

    @classmethod
    @abc.abstractmethod
    def make_sampler_fn(cls, **kwargs) -> TaskSampler:
        """Create the TaskSampler given keyword arguments.

        These `kwargs` will be generated by one of
        `ExperimentConfig.train_task_sampler_args`,
        `ExperimentConfig.valid_task_sampler_args`, or
        `ExperimentConfig.test_task_sampler_args` depending on whether
        the user has chosen to train, validate, or test.
        """
        raise NotImplementedError()

    def train_task_sampler_args(
        self,
        process_ind: int,
        total_processes: int,
        devices: Optional[List[int]] = None,
        seeds: Optional[List[int]] = None,
        deterministic_cudnn: bool = False,
    ) -> Dict[str, Any]:
        """Specifies the training parameters for the `process_ind`th training
        process.

        These parameters are meant be passed as keyword arguments to `ExperimentConfig.make_sampler_fn`
        to generate a task sampler.

        # Parameters

        process_ind : The unique index of the training process (`0 ≤ process_ind < total_processes`).
        total_processes : The total number of training processes.
        devices : Gpu devices (if any) to use.
        seeds : The seeds to use, if any.
        deterministic_cudnn : Whether or not to use deterministic cudnn.

        # Returns

        The parameters for `make_sampler_fn`
        """
        raise NotImplementedError()

    def valid_task_sampler_args(
        self,
        process_ind: int,
        total_processes: int,
        devices: Optional[List[int]] = None,
        seeds: Optional[List[int]] = None,
        deterministic_cudnn: bool = False,
    ) -> Dict[str, Any]:
        """Specifies the validation parameters for the `process_ind`th
        validation process.

        See `ExperimentConfig.train_task_sampler_args` for parameter
        definitions.
        """
        raise NotImplementedError()

    def test_task_sampler_args(
        self,
        process_ind: int,
        total_processes: int,
        devices: Optional[List[int]] = None,
        seeds: Optional[List[int]] = None,
        deterministic_cudnn: bool = False,
    ) -> Dict[str, Any]:
        """Specifies the test parameters for the `process_ind`th test process.

        See `ExperimentConfig.train_task_sampler_args` for parameter
        definitions.
        """
        raise NotImplementedError()
