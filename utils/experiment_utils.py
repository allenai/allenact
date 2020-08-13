"""Utility classes and functions for running and designing experiments."""
import abc
import collections.abc
import copy
import random
import typing
from collections import OrderedDict
from typing import (
    Callable,
    NamedTuple,
    Dict,
    Any,
    Union,
    Iterator,
    Optional,
    List,
    Tuple,
    cast,
)

import numpy as np
import torch
from torch import optim

from core.algorithms.onpolicy_sync.losses.abstract_loss import AbstractActorCriticLoss
from core.base_abstractions.misc import Loss


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
        return typing.cast(Callable, self.class_type)(**allkwargs)


class ScalarMeanTracker(object):
    """Track a collection `scalar key -> mean` pairs."""

    def __init__(self) -> None:
        self._sums: Dict[str, float] = OrderedDict()
        self._counts: Dict[str, int] = OrderedDict()

    def add_scalars(self, scalars: Dict[str, Union[float, int]], n: int = 1) -> None:
        """Add additional scalars to track.

        # Parameters

        scalars : A dictionary of `scalar key -> value` pairs.
        """
        for k in scalars:
            if k not in self._sums:
                self._sums[k] = n * scalars[k]
                self._counts[k] = n
            else:
                self._sums[k] += n * scalars[k]
                self._counts[k] += n

    def pop_and_reset(self) -> Dict[str, float]:
        """Return tracked means and reset.

        On resetting all previously tracked values are discarded.

        # Returns

        A dictionary of `scalar key -> current mean` pairs corresponding to those
        values added with `add_scalars`.
        """
        means = OrderedDict(
            [(k, float(self._sums[k] / self._counts[k])) for k in self._sums]
        )
        self.reset()
        return means

    def reset(self):
        self._sums = OrderedDict()
        self._counts = OrderedDict()

    def sums(self):
        return copy.copy(self._sums)

    def counts(self):
        return copy.copy(self._counts)

    def means(self):
        return OrderedDict(
            [(k, float(self._sums[k] / self._counts[k])) for k in self._sums]
        )

    @property
    def empty(self):
        assert len(self._sums) == len(
            self._counts
        ), "Mismatched length of _sums {} and _counts {}".format(
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


# noinspection PyTypeHints,PyUnresolvedReferences
def set_deterministic_cudnn() -> None:
    """Makes cudnn deterministic.

    This may slow down computations.
    """
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True  # type: ignore
        torch.backends.cudnn.benchmark = False  # type: ignore


def set_seed(seed: Optional[int] = None) -> None:
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


class EarlyStoppingCriterion(abc.ABC):
    """Abstract class for class who determines if training should stop early in
    a particular pipeline stage."""

    @abc.abstractmethod
    def __call__(
        self,
        stage_steps: int,
        total_steps: int,
        training_metrics: ScalarMeanTracker,
        test_valid_metrics: List[Tuple[str, int, Union[float, np.ndarray]]],
    ) -> bool:
        """Returns `True` if training should be stopped early.

        # Parameters

        stage_steps: Total number of steps taken in the current pipeline stage.
        total_steps: Total number of steps taken during training so far (includes steps
            taken in prior pipeline stages).
        training_metrics: Metrics recovered over some fixed number of steps
            (see the `metric_accumulate_interval` attribute in the `TrainingPipeline` class)
            training.
        test_valid_metrics: A tuple `(key, steps, value)` where key is the metric's name
             prefixed by either `"valid/"` or `"test/"`, `steps` is the total number of
             steps that the validation/test model was trained for, and value is the
             value of the metric.
        """
        raise NotImplementedError


class NeverEarlyStoppingCriterion(EarlyStoppingCriterion):
    """Implementation of `EarlyStoppingCriterion` which never stops early."""

    def __call__(
        self,
        stage_steps: int,
        total_steps: int,
        training_metrics: ScalarMeanTracker,
        test_valid_metrics: List[Tuple[str, int, Union[float, np.ndarray]]],
    ) -> bool:
        return False


class OffPolicyPipelineComponent(NamedTuple):
    data_iterator_builder: Callable[[], Iterator]
    loss_names: List[str]
    updates: int
    loss_weights: Optional[typing.Sequence[float]] = None


class PipelineStage(object):
    """A single stage in a training pipeline.

    # Attributes

    loss_name : A collection of unique names assigned to losses. These will
        reference the `Loss` objects in a `TrainingPipeline` instance.
    max_stage_steps : Either the total number of steps agents should take in this stage or
        a Callable object (e.g. a function)
    early_stopping_criterion: An `EarlyStoppingCriterion` object which determines if
        training in this stage should be stopped early. If `None` then no early stopping
        occurs. If `early_stopping_criterion` is not `None` then we do not guarantee
        reproducibility when restarting a model from a checkpoint (as the
         `EarlyStoppingCriterion` object may store internal state which is not
         saved in the checkpoint).
    loss_weights : A list of floating point numbers describing the relative weights
        applied to the losses referenced by `loss_name`. Should be the same length
        as `loss_name`. If this is `None`, all weights will be assumed to be one.
    teacher_forcing : If applicable, defines the probability an agent will take the
        expert action (as opposed to its own sampled action) at a given time point.
    """

    def __init__(
        self,
        loss_names: List[str],
        max_stage_steps: Union[int, Callable],
        early_stopping_criterion: Optional[EarlyStoppingCriterion] = None,
        loss_weights: Optional[typing.Sequence[float]] = None,
        teacher_forcing: Optional[LinearDecay] = None,
        offpolicy_component: Optional[OffPolicyPipelineComponent] = None,
    ):
        self.loss_names = loss_names
        self.max_stage_steps = max_stage_steps
        self.early_stopping_criterion = early_stopping_criterion
        self.loss_weights = loss_weights
        self.teacher_forcing = teacher_forcing
        self.offpolicy_component = offpolicy_component

        self.steps_taken_in_stage: int = 0
        self.rollout_count = 0
        self.early_stopping_criterion_met = False

        self.named_losses: Optional[Dict[str, AbstractActorCriticLoss]] = None
        self._named_loss_weights: Optional[Dict[str, float]] = None

    @property
    def is_complete(self):
        return (
            self.early_stopping_criterion_met
            or self.steps_taken_in_stage >= self.max_stage_steps
        )

    @property
    def named_loss_weights(self):
        if self._named_loss_weights is None:
            loss_weights = (
                self.loss_weights
                if self.loss_weights is not None
                else [1.0] * len(self.loss_names)
            )
            self._named_loss_weights = {
                name: weight for name, weight in zip(self.loss_names, loss_weights)
            }
        return self._named_loss_weights


class TrainingPipeline(object):
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
    save_interval : The frequency with which to save (in total agent steps taken). If `None` then *no*
        checkpoints will be saved. Otherwise, in addition to the checkpoints being saved every
        `save_interval` steps, a checkpoint will *always* be saved at the end of each pipeline stage.
        If `save_interval <= 0` then checkpoints will only be saved at the end of each pipeline stage.
    metric_accumulate_interval : The frequency with which training/validation metrics are accumulated
        (in total agent steps). Metrics accumulated in an interval are logged (if `should_log` is `True`)
        and used by the stage's early stopping criterion (if any).
    should_log: `True` if metrics accumulated during training should be logged to the console as well
        as to a tensorboard file.
    lr_scheduler_builder : Optional builder object to instantiate the learning rate scheduler used
        through the pipeline.
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
        save_interval: Optional[int],
        metric_accumulate_interval: int,
        should_log: bool = True,
        lr_scheduler_builder: Optional[Builder[optim.lr_scheduler._LRScheduler]] = None,  # type: ignore
    ):
        """Initializer.

        See class docstring for parameter definitions.
        """
        self.save_interval = save_interval
        self.metric_accumulate_interval = metric_accumulate_interval

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
        self.should_log = should_log

        self.pipeline_stages = pipeline_stages

        self._current_stage: Optional[PipelineStage] = None

        self.backprop_count = 0
        self.rollout_count = 0
        self.off_policy_epochs = None

        self._refresh_current_stage(force_stage_search_from_start=True)

    @property
    def total_steps(self) -> int:
        return sum(ps.steps_taken_in_stage for ps in self.pipeline_stages)

    def _refresh_current_stage(
        self, force_stage_search_from_start: bool = False
    ) -> Optional[PipelineStage]:
        if force_stage_search_from_start:
            self._current_stage = None

        if self._current_stage is None or self._current_stage.is_complete:
            if self._current_stage is None:
                start_index = 0
            else:
                start_index = self.pipeline_stages.index(self._current_stage) + 1

            self._current_stage = None
            for ps in self.pipeline_stages[start_index:]:
                if not ps.is_complete:
                    self._current_stage = ps
                    break
        return self._current_stage

    @property
    def current_stage(self) -> Optional[PipelineStage]:
        return self._current_stage

    @property
    def current_stage_index(self) -> Optional[int]:
        if self.current_stage is None:
            return None
        return self.pipeline_stages.index(self.current_stage)

    def before_rollout(self, train_valid_metrics: Optional[Dict] = None):
        if train_valid_metrics is not None:
            self.current_stage.early_stopping_criterion_met = self.current_stage.early_stopping_criterion(
                stage_steps=self.current_stage.steps_taken_in_stage,
                total_steps=self.total_steps,
                training_metrics=train_valid_metrics["train"],
                test_valid_metrics=train_valid_metrics["valid"],
            )
        self._refresh_current_stage(force_stage_search_from_start=False)

    def restart_pipeline(self):
        for ps in self.pipeline_stages:
            ps.steps_taken_in_stage = 0
            ps.early_stopping_criterion_met = False
        self._current_stage = None
        self._refresh_current_stage(force_stage_search_from_start=True)

    def state_dict(self):
        return dict(
            stage_info_list=[
                {
                    "early_stopping_criterion_met": ps.early_stopping_criterion_met,
                    "steps_taken_in_stage": ps.steps_taken_in_stage,
                }
                for ps in self.pipeline_stages
            ],
            rollout_count=self.rollout_count,
            backprop_count=self.backprop_count,
            off_policy_epochs=self.off_policy_epochs,
        )

    def load_state_dict(self, state_dict: Dict[str, Any]):
        for ps, stage_info in zip(self.pipeline_stages, state_dict["stage_info_list"]):
            ps.early_stopping_criterion_met = stage_info["early_stopping_criterion_met"]
            ps.steps_taken_in_stage = stage_info["steps_taken_in_stage"]

        self.rollout_count = state_dict["rollout_count"]
        self.backprop_count = state_dict["backprop_count"]
        self.off_policy_epochs = state_dict["off_policy_epochs"]

        self._refresh_current_stage(force_stage_search_from_start=True)

    @property
    def current_stage_losses(self) -> Dict[str, AbstractActorCriticLoss]:
        if self.current_stage.named_losses is None:
            for loss_name in self.current_stage.loss_names:
                if isinstance(self.named_losses[loss_name], Builder):
                    self.named_losses[loss_name] = typing.cast(
                        Builder["AbstractActorCriticLoss"],
                        self.named_losses[loss_name],
                    )()

            self.current_stage.named_losses = {
                loss_name: cast(AbstractActorCriticLoss, self.named_losses[loss_name])
                for loss_name in self.current_stage.loss_names
            }
        return self.current_stage.named_losses

    @property
    def current_stage_loss_weights(self) -> Dict[str, float]:
        return self.current_stage.named_loss_weights
