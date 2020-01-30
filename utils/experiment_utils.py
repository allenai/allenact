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
    def __init__(self) -> None:
        self._sums: Dict[str, float] = {}
        self._counts: Dict[str, int] = {}

    def add_scalars(self, scalars):
        for k in scalars:
            if k not in self._sums:
                self._sums[k] = scalars[k]
                self._counts[k] = 1
            else:
                self._sums[k] += scalars[k]
                self._counts[k] += 1

    def pop_and_reset(self):
        means = {k: self._sums[k] / self._counts[k] for k in self._sums}
        self._sums = {}
        self._counts = {}
        return means


class LinearDecay:
    def __init__(self, steps: int, startp: float = 1.0, endp: float = 0.0) -> None:
        self.steps = steps
        self.startp = startp
        self.endp = endp

    def __call__(self, epoch: int) -> float:
        epoch = max(min(epoch, self.steps), 0)
        return self.startp + (self.endp - self.startp) * (epoch / float(self.steps))


def set_deterministic_cudnn():
    if torch.cuda.is_available():
        # noinspection PyUnresolvedReferences
        torch.backends.cudnn.deterministic = True
        # noinspection PyUnresolvedReferences
        torch.backends.cudnn.benchmark = False


def set_seed(seed: int):
    torch.manual_seed(seed)  # seeds the RNG for all devices (CPU and GPUs)
    random.seed(seed)
    np.random.seed(seed)


class PipelineStage(NamedTuple):
    loss_names: typing.List[str]
    end_criterion: int
    loss_weights: Optional[typing.Sequence[float]] = None
    teacher_forcing: Optional[LinearDecay] = None


class TrainingPipeline(Iterator):
    # noinspection PyUnresolvedReferences
    def __init__(
        self,
        named_losses: Optional[Dict[str, Union[Loss, Builder[Loss]]]],
        pipeline_stages: List[PipelineStage],
        optimizer: Union[optim.Optimizer, Builder[optim.Optimizer]],  # type: ignore
        num_mini_batch: int,
        update_repeats: int,
        max_grad_norm: float,
        num_steps: int,
        gamma: float,
        use_gae: bool,
        gae_lambda: float,
        save_interval: int,
        log_interval: int,
    ):
        self.save_interval = save_interval
        self.log_interval = log_interval

        self.optimizer = optimizer
        self.num_mini_batch = num_mini_batch

        self.update_repeats = update_repeats
        self.max_grad_norm = max_grad_norm
        self.num_steps = num_steps
        self.named_losses = named_losses
        self.gamma = gamma
        self.use_gae = use_gae
        self.gae_lambda = gae_lambda

        self.pipeline_stages = pipeline_stages

        self.current_pipeline_stage = -1

    def current_stage(self):
        return (
            None
            if (
                len(self.pipeline_stages) <= self.current_pipeline_stage
                or self.current_pipeline_stage < 0
            )
            else self.pipeline_stages[self.current_pipeline_stage]
        )

    def reset(self):
        self.current_pipeline_stage = -1

    def __next__(self):
        if len(self.pipeline_stages) == self.current_pipeline_stage + 1:
            raise StopIteration()

        self.current_pipeline_stage += 1

        return self.pipeline_stages[self.current_pipeline_stage]
