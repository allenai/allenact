"""Utility classes and functions for running and designing experiments."""

import collections.abc
import copy
import random
import typing
from typing import NamedTuple, Dict, Any, Union, Iterator, Optional, List

import numpy as np
import torch
from torch import optim

from rl_base.common import Loss


def recursive_update(
    original: Union[Dict, collections.abc.MutableMapping],
    update: Union[Dict, collections.abc.MutableMapping],
):
    """Recursively updates original dictionary with entries form update dict.

    # Parameters

    original : Original dictionary to be updated.
    update : Dictionary with additional or replacement entries.

    # Returns

    Updated original dictionary.
    """
    for k, v in update.items():
        if isinstance(v, collections.abc.MutableMapping):
            original[k] = recursive_update(original.get(k, {}), v)
        else:
            original[k] = v
    return original


ToBuildType = typing.TypeVar("ToBuildType")


class Builder(tuple, typing.Generic[ToBuildType]):
    """Used to instantiate a given class with (default) parameters.

    Helper class that stores a class, default parameters for that
    class, and key word arguments that (possibly) overwrite the defaults.
    When calling this an object of the Builder class it generates
    a class of type `class_type` with parameters specified by
    the attributes `default` and `kwargs` (and possibly additional, overwriting,
    keyword arguments).

    # Attributes

    class_type : The class to be instantiated when calling the object.
    kwargs : Keyword arguments used to instantiate an object of type `class_type`.
    default : Default parameters used when instantiating the class.
    """

    class_type: ToBuildType
    kwargs: Dict[str, Any]
    default: Dict[str, Any]

    # noinspection PyTypeChecker
    def __new__(
        cls,
        class_type: ToBuildType,
        kwargs: Optional[Dict[str, Any]] = None,
        default: Optional[Dict[str, Any]] = None,
    ):
        """Create a new Builder.

        For parameter descriptions see the class documentation. Note
        that `kwargs` and `default` can be None in which case they are
        set to be empty dictionaries.
        """
        self = tuple.__new__(
            cls,
            (
                class_type,
                kwargs if kwargs is not None else {},
                default if default is not None else {},
            ),
        )
        self.class_type = class_type
        self.kwargs = self[1]
        self.default = self[2]
        return self

    def __repr__(self) -> str:
        return (
            f"Group(class_type={self.class_type},"
            f" kwargs={self.kwargs},"
            f" default={self.default})"
        )

    def __call__(self, **kwargs) -> ToBuildType:
        """Build and return a new class.

        # Parameters
        kwargs : additional keyword arguments to use when instantiating
            the object. These overwrite all arguments already in the `self.kwargs`
            and `self.default` attributes.

        # Returns

        Class of type `self.class_type` with parameters
        taken from `self.default`, `self.kwargs`, and
        any keyword arguments additionally passed to `__call__`.
        """
        allkwargs = copy.deepcopy(self.default)
        recursive_update(allkwargs, self.kwargs)
        recursive_update(allkwargs, kwargs)
        return typing.cast(typing.Callable, self.class_type)(**allkwargs)


class ScalarMeanTracker(object):
    """Track a collection `scalar key -> mean` pairs."""

    def __init__(self) -> None:
        self._sums: Dict[str, float] = {}
        self._counts: Dict[str, int] = {}

    def add_scalars(self, scalars: Dict[str, Union[float, int]]) -> None:
        """Add additional scalars to track.

        # Parameters

        scalars : A dictionary of `scalar key -> value` pairs.
        """
        for k in scalars:
            if k not in self._sums:
                self._sums[k] = scalars[k]
                self._counts[k] = 1
            else:
                self._sums[k] += scalars[k]
                self._counts[k] += 1

    def pop_and_reset(self) -> Dict[str, float]:
        """Return tracked means and reset.

        On resetting all previously tracked values are discarded.

        # Returns

        A dictionary of `scalar key -> current mean` pairs corresponding to those
        values added with `add_scalars`.
        """
        means = {k: float(self._sums[k] / self._counts[k]) for k in self._sums}
        self._sums = {}
        self._counts = {}
        return means

    @property
    def empty(self):
        assert len(self._sums) == len(self._counts), "Mismatched length of _sums {} and _counts {}".format(
            len(self._sums), len(self._counts)
        )
        return len(self._sums) == 0


class LinearDecay(object):
    """Linearly decay between two values over some number of steps.

    Obtain the value corresponding to the `i`th step by calling
    an instantiation of this object with the value `i`.

    # Parameters

    steps : The number of steps over which to decay.
    startp : The starting value.
    endp : The ending value.
    """

    def __init__(self, steps: int, startp: float = 1.0, endp: float = 0.0) -> None:
        """Initializer.

        See class documentation for parameter definitions.
        """
        self.steps = steps
        self.startp = startp
        self.endp = endp

    def __call__(self, epoch: int) -> float:
        """Get the decayed value for `epoch` number of steps.

        # Parameters

        epoch : The number of steps.

        # Returns

        Decayed value for `epoch` number of steps.
        """
        epoch = max(min(epoch, self.steps), 0)
        return self.startp + (self.endp - self.startp) * (epoch / float(self.steps))


def set_deterministic_cudnn() -> None:
    """Makes cudnn deterministic.

    This may slow down computations.
    """
    if torch.cuda.is_available():
        # noinspection PyUnresolvedReferences
        torch.backends.cudnn.deterministic = True  # type: ignore
        # noinspection PyUnresolvedReferences
        torch.backends.cudnn.benchmark = False  # type: ignore


def set_seed(seed: Optional[int]=None) -> None:
    """Set seeds for multiple (cpu) sources of randomness.

    Sets seeds for (cpu) `pytorch`, base `random`, and `numpy`.

    # Parameters

    seed : The seed to set. If set to None, keep using the current seed.
    """
    if seed is None:
        return

    torch.manual_seed(seed)  # seeds the RNG for all devices (CPU and GPUs)
    random.seed(seed)
    np.random.seed(seed)


class PipelineStage(NamedTuple):
    """A single stage in a training pipeline.

    # Attributes

    loss_name : A collection of unique names assigned to losses. These will
        reference the `Loss` objects in a `TrainingPipeline` instance.
    end_criterion : Total number of steps agents should take in this stage.
    loss_weights : A list of floating point numbers describing the relative weights
        applied to the losses referenced by `loss_name`. Should be the same length
        as `loss_name`. If this is `None`, all weights will be assumed to be one.
    teacher_forcing : If applicable, defines the probability an agent will take the
        expert action (as opposed to its own sampeld action) at a given time point.
    """

    loss_names: typing.List[str]
    end_criterion: int
    loss_weights: Optional[typing.Sequence[float]] = None
    teacher_forcing: Optional[LinearDecay] = None


class TrainingPipeline(typing.Iterable):
    """Class defining the stages (and global parameters) in a training
    pipeline.

    The training pipeline can be used as an iterator to go through the pipeline
    stages in, for instance, a loop.

    # Attributes

    named_losses : Dictionary mapping a the name of a loss to either an instantiation
        of that loss or a `Builder` that, when called, will return that loss.
    pipeline_stages : A list of PipelineStages. Each of these define how the agent
        will be trained and are executed sequentially.
    optimizer_builder : Builder object to instantiate the optimizer to use during training.
    num_mini_batch : The number of mini-batches to break a rollout into.
    update_repeats : The number of times we will cycle through the mini-batches corresponding
        to a single rollout doing gradient updates.
    max_grad_norm : The maximum "inf" norm of any gradient step (gradients are clipped to not exceed this).
    num_steps : Total number of steps a single agent takes in a rollout.
    gamma : Discount factor applied to rewards (should be in [0, 1]).
    use_gae : Whether or not to use generalized advantage estimation (GAE).
    gae_lambda : The additional parameter used in GAE.
    save_interval : The frequency with which to save (in total agent steps).
    log_interval : The frequency with which to log metrics to tensorboard (in total agent steps).
    current_pipeline_stage : Integer tracking the current stage of the pipeline. If -1 then the pipeline
        is at it's start and `__next__` will need to be called to get the first pipeline stage.
    lr_scheduler_builder : Optional builder object to instantiate the learning rate scheduler used through the pipeline.
    """

    # noinspection PyUnresolvedReferences
    def __init__(
        self,
        named_losses: Dict[str, Union[Loss, Builder[Loss]]],
        pipeline_stages: List[PipelineStage],
        optimizer_builder: Builder[optim.Optimizer],  # type: ignore
        num_mini_batch: int,
        update_repeats: int,
        max_grad_norm: float,
        num_steps: int,
        gamma: float,
        use_gae: bool,
        gae_lambda: float,
        advance_scene_rollout_period: Optional[int],
        save_interval: int,
        log_interval: int,
        lr_scheduler_builder: Optional[Builder[optim.lr_scheduler._LRScheduler]] = None,  # type: ignore
    ):
        """Initializer.

        See class docstring for parameter definitions.
        """
        self.save_interval = save_interval
        self.log_interval = log_interval

        self.optimizer_builder = optimizer_builder
        self.lr_scheduler_builder = lr_scheduler_builder
        self.num_mini_batch = num_mini_batch

        self.update_repeats = update_repeats
        self.max_grad_norm = max_grad_norm
        self.num_steps = num_steps
        self.named_losses = named_losses
        self.gamma = gamma
        self.use_gae = use_gae
        self.gae_lambda = gae_lambda
        self.advance_scene_rollout_period = advance_scene_rollout_period

        self.pipeline_stages = pipeline_stages

    def __iter__(self) -> Iterator[typing.Tuple[int, PipelineStage]]:
        """Create iterator which moves through the pipeline stages."""
        return enumerate(self.pipeline_stages)

    def iterator_starting_at(
        self, start_stage_num: int
    ) -> Iterator[typing.Tuple[int, PipelineStage]]:
        """Create iterator which moves through the pipeline stages starting at
        stage `start_stage_num`."""
        return zip(
            range(start_stage_num, len(self.pipeline_stages)),
            self.pipeline_stages[start_stage_num:],
        )
