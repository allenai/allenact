"""Utility classes and functions for running and designing experiments."""
import abc
import collections.abc
import copy
import random
from collections import OrderedDict, defaultdict
from typing import (
    Callable,
    NamedTuple,
    Dict,
    Any,
    Union,
    Iterator,
    Optional,
    List,
    cast,
    Sequence,
    TypeVar,
    Generic,
)

import numpy as np
import torch
import torch.optim as optim

from allenact.algorithms.offpolicy_sync.losses.abstract_offpolicy_loss import (
    AbstractOffPolicyLoss,
    Memory,
)
from allenact.algorithms.onpolicy_sync.losses.abstract_loss import (
    AbstractActorCriticLoss,
)
from allenact.algorithms.onpolicy_sync.storage import ExperienceStorage
from allenact.base_abstractions.misc import Loss
from allenact.utils.misc_utils import prepare_locals_for_super
from allenact.utils.system import get_logger


def evenly_distribute_count_into_bins(count: int, nbins: int) -> List[int]:
    """Distribute a count into a number of bins.

    # Parameters
    count: A positive integer to be distributed, should be `>= nbins`.
    nbins: The number of bins.

    # Returns
    A list of positive integers which sum to `count`. These values will be
    as close to equal as possible (may differ by at most 1).
    """
    assert count >= nbins, f"count ({count}) < nbins ({nbins})"
    res = [0] * nbins
    for it in range(count):
        res[it % nbins] += 1
    return res


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


ToBuildType = TypeVar("ToBuildType")


class Builder(tuple, Generic[ToBuildType]):
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
        return cast(Callable, self.class_type)(**allkwargs)


class ScalarMeanTracker(object):
    """Track a collection `scalar key -> mean` pairs."""

    def __init__(self) -> None:
        self._sums: Dict[str, float] = OrderedDict()
        self._counts: Dict[str, int] = OrderedDict()

    def add_scalars(
        self, scalars: Dict[str, Union[float, int]], n: Union[int, Dict[str, int]] = 1
    ) -> None:
        """Add additional scalars to track.

        # Parameters

        scalars : A dictionary of `scalar key -> value` pairs.
        """
        ndict = cast(
            Dict[str, int], (n if isinstance(n, Dict) else defaultdict(lambda: n))  # type: ignore
        )

        for k in scalars:
            if k not in self._sums:
                self._sums[k] = ndict[k] * scalars[k]
                self._counts[k] = ndict[k]
            else:
                self._sums[k] += ndict[k] * scalars[k]
                self._counts[k] += ndict[k]

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

    def counts(self) -> Dict[str, int]:
        return copy.copy(self._counts)

    def means(self) -> Dict[str, float]:
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


class LoggingPackage(object):
    """Data package used for logging."""

    def __init__(
        self,
        mode: str,
        training_steps: Optional[int],
        pipeline_stage: Optional[int] = None,
        off_policy_steps: Optional[int] = None,
    ) -> None:
        self.mode = mode

        self.training_steps: int = training_steps
        self.pipeline_stage = pipeline_stage
        self.off_policy_steps: Optional[int] = off_policy_steps

        self.metrics_tracker = ScalarMeanTracker()
        self.train_info_tracker = ScalarMeanTracker()
        self.metric_dicts: List[Any] = []
        self.viz_data: Optional[Dict[str, List[Dict[str, Any]]]] = None
        self.checkpoint_file_name: Optional[str] = None

        self.num_empty_metrics_dicts_added: int = 0

    @property
    def num_non_empty_metrics_dicts_added(self) -> int:
        return len(self.metric_dicts)

    @staticmethod
    def _metrics_dict_is_empty(
        single_task_metrics_dict: Dict[str, Union[float, int]]
    ) -> bool:
        return (
            len(single_task_metrics_dict) == 0
            or (
                len(single_task_metrics_dict) == 1
                and "task_info" in single_task_metrics_dict
            )
            or (
                "success" in single_task_metrics_dict
                and single_task_metrics_dict["success"] is None
            )
        )

    def add_metrics_dict(
        self, single_task_metrics_dict: Dict[str, Union[float, int]]
    ) -> bool:
        if self._metrics_dict_is_empty(single_task_metrics_dict):
            self.num_empty_metrics_dicts_added += 1
            return False

        self.metric_dicts.append(single_task_metrics_dict)
        self.metrics_tracker.add_scalars(
            {k: v for k, v in single_task_metrics_dict.items() if k != "task_info"}
        )
        return True

    def add_train_info_dict(
        self, train_info_dict: Dict[str, Union[int, float]], n: int
    ):
        assert n >= 0
        self.train_info_tracker.add_scalars(scalars=train_info_dict, n=n)


class LinearDecay(object):
    """Linearly decay between two values over some number of steps.

    Obtain the value corresponding to the `i`-th step by calling
    an instance of this class with the value `i`.

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


class MultiLinearDecay(object):
    """Container for multiple stages of LinearDecay.

    Obtain the value corresponding to the `i`-th step by calling
    an instance of this class with the value `i`.

    # Parameters

    stages: List of `LinearDecay` objects to be sequentially applied
        for the number of steps in each stage.
    """

    def __init__(self, stages: Sequence[LinearDecay]) -> None:
        """Initializer.

        See class documentation for parameter definitions.
        """
        self.stages = stages
        self.steps = np.cumsum([stage.steps for stage in self.stages])
        self.total_steps = self.steps[-1]
        self.stage_idx = -1
        self.min_steps = 0
        self.max_steps = 0
        self.stage = None

    def __call__(self, epoch: int) -> float:
        """Get the decayed value factor for `epoch` number of steps.

        # Parameters

        epoch : The number of steps.

        # Returns

        Decayed value for `epoch` number of steps.
        """
        epoch = max(min(epoch, self.total_steps), 0)

        while epoch >= self.max_steps and self.max_steps < self.total_steps:
            self.stage_idx += 1
            assert self.stage_idx < len(self.stages)

            self.min_steps = self.max_steps
            self.max_steps = self.steps[self.stage_idx]
            self.stage = self.stages[self.stage_idx]

        return self.stage(epoch - self.min_steps)


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
        self, stage_steps: int, total_steps: int, training_metrics: ScalarMeanTracker,
    ) -> bool:
        """Returns `True` if training should be stopped early.

        # Parameters

        stage_steps: Total number of steps taken in the current pipeline stage.
        total_steps: Total number of steps taken during training so far (includes steps
            taken in prior pipeline stages).
        training_metrics: Metrics recovered over some fixed number of steps
            (see the `metric_accumulate_interval` attribute in the `TrainingPipeline` class)
            training.
        """
        raise NotImplementedError


class NeverEarlyStoppingCriterion(EarlyStoppingCriterion):
    """Implementation of `EarlyStoppingCriterion` which never stops early."""

    def __call__(
        self, stage_steps: int, total_steps: int, training_metrics: ScalarMeanTracker,
    ) -> bool:
        return False


class OffPolicyPipelineComponent(NamedTuple):
    """An off-policy component for a PipeLineStage.

    # Attributes

    data_iterator_builder: A function to instantiate a Data Iterator (with a __next__(self) method)
    loss_names: list of unique names assigned to off-policy losses
    updates: number of off-policy updates between on-policy rollout collections
    loss_weights : A list of floating point numbers describing the relative weights
        applied to the losses referenced by `loss_names`. Should be the same length
        as `loss_names`. If this is `None`, all weights will be assumed to be one.
    data_iterator_kwargs_generator: Optional generator of keyword arguments for data_iterator_builder (useful for
        distributed training. It takes
        a `cur_worker` int value,
        a `rollouts_per_worker` list of number of samplers per training worker,
        and an optional random `seed` shared by all workers, which can be None.
    """

    data_iterator_builder: Callable[..., Iterator]
    loss_names: List[str]
    updates: int
    loss_weights: Optional[Sequence[float]] = None
    data_iterator_kwargs_generator: Callable[
        [int, Sequence[int], Optional[int]], Dict
    ] = lambda cur_worker, rollouts_per_worker, seed: {}


class CustomPipelineComponent(NamedTuple):
    """An custom component for a PipeLineStage.

    # Attributes

    uuid: the name of this component
    storage_name: the name of the `ExperienceStorage` that will be used with this component.
    loss_names: list of unique names assigned to off-policy losses
    num_mini_batch:
    update_repeats:
    loss_weights : A list of floating point numbers describing the relative weights
        applied to the losses referenced by `loss_names`. Should be the same length
        as `loss_names`. If this is `None`, all weights will be assumed to be one.
    """

    uuid: str
    storage_name: str
    loss_names: List[str]
    num_mini_batch: Optional[int]
    update_repeats: Optional[int]
    loss_weights: Optional[Sequence[float]] = None


class TrainingSettings(object):
    """Class defining parameters used for training (within a stage or the
    entire pipeline).

    # Attributes

    num_mini_batch : The number of mini-batches to break a rollout into.
    update_repeats : The number of times we will cycle through the mini-batches corresponding
        to a single rollout doing gradient updates.
    max_grad_norm : The maximum "inf" norm of any gradient step (gradients are clipped to not exceed this).
    num_steps : Total number of steps a single agent takes in a rollout.
    gamma : Discount factor applied to rewards (should be in [0, 1]).
    use_gae : Whether or not to use generalized advantage estimation (GAE).
    gae_lambda : The additional parameter used in GAE.
    advance_scene_rollout_period: Optional number of rollouts before enforcing an advance scene in all samplers.
    save_interval : The frequency with which to save (in total agent steps taken). If `None` then *no*
        checkpoints will be saved. Otherwise, in addition to the checkpoints being saved every
        `save_interval` steps, a checkpoint will *always* be saved at the end of each pipeline stage.
        If `save_interval <= 0` then checkpoints will only be saved at the end of each pipeline stage.
    metric_accumulate_interval : The frequency with which training/validation metrics are accumulated
        (in total agent steps). Metrics accumulated in an interval are logged (if `should_log` is `True`)
        and used by the stage's early stopping criterion (if any).
    """

    num_mini_batch: Optional[int]
    update_repeats: Optional[int]
    max_grad_norm: Optional[float]
    num_steps: Optional[int]
    gamma: Optional[float]
    use_gae: Optional[bool]
    gae_lambda: Optional[float]
    advance_scene_rollout_period: Optional[int]
    save_interval: Optional[int]
    metric_accumulate_interval: Optional[int]

    # noinspection PyUnresolvedReferences
    def __init__(
        self,
        num_mini_batch: Optional[int] = None,
        update_repeats: Optional[int] = None,
        max_grad_norm: Optional[float] = None,
        num_steps: Optional[int] = None,
        gamma: Optional[float] = None,
        use_gae: Optional[bool] = None,
        gae_lambda: Optional[float] = None,
        advance_scene_rollout_period: Optional[int] = None,
        save_interval: Optional[int] = None,
        metric_accumulate_interval: Optional[int] = None,
        **kwargs: Any,
    ):
        all_vars = prepare_locals_for_super(locals(), ignore_kwargs=True)

        for key, value in all_vars.items():
            setattr(self, key, value)


_TRAINING_SETTINGS_NAMES: List[str] = list(TrainingSettings().__dict__.keys())


class PipelineStage(TrainingSettings):
    """A single stage in a training pipeline, possibly including overrides to
    the global `TrainingSettings` in `TrainingPipeline`.

    # Attributes

    loss_name : A collection of unique names assigned to losses. These will
        reference the `Loss` objects in a `TrainingPipeline` instance.
    max_stage_steps : Either the total number of steps agents should take in this stage or
        a Callable object (e.g. a function)
    loss_weights : A list of floating point numbers describing the relative weights
        applied to the losses referenced by `loss_name`. Should be the same length
        as `loss_name`. If this is `None`, all weights will be assumed to be one.
    teacher_forcing : If applicable, defines the probability an agent will take the
        expert action (as opposed to its own sampled action) at a given time point.
    early_stopping_criterion: An `EarlyStoppingCriterion` object which determines if
        training in this stage should be stopped early. If `None` then no early stopping
        occurs. If `early_stopping_criterion` is not `None` then we do not guarantee
        reproducibility when restarting a model from a checkpoint (as the
        `EarlyStoppingCriterion` object may store internal state which is not
        saved in the checkpoint). Currently AllenAct only supports using early stopping
        criterion when **not** using distributed training.
    num_mini_batch : See docs for `TrainingSettings`.
    update_repeats : See docs for `TrainingSettings`.
    max_grad_norm : See docs for `TrainingSettings`.
    num_steps : See docs for `TrainingSettings`.
    gamma : See docs for `TrainingSettings`.
    use_gae : See docs for `TrainingSettings`.
    gae_lambda : See docs for `TrainingSettings`.
    advance_scene_rollout_period: See docs for `TrainingSettings`.
    save_interval : See docs for `TrainingSettings`.
    metric_accumulate_interval : See docs for `TrainingSettings`.
    """

    def __init__(
        self,
        *,  # Disables positional arguments. Please provide arguments as keyword arguments.
        loss_names: List[str],
        max_stage_steps: Union[int, Callable],
        loss_weights: Optional[Sequence[float]] = None,
        loss_update_repeats: Optional[Sequence[int]] = None,
        teacher_forcing: Optional[LinearDecay] = None,
        offpolicy_component: Optional[OffPolicyPipelineComponent] = None,
        custom_components: Optional[Sequence[CustomPipelineComponent]] = None,
        early_stopping_criterion: Optional[EarlyStoppingCriterion] = None,
        num_mini_batch: Optional[int] = None,
        update_repeats: Optional[int] = None,
        max_grad_norm: Optional[float] = None,
        num_steps: Optional[int] = None,
        gamma: Optional[float] = None,
        use_gae: Optional[bool] = None,
        gae_lambda: Optional[float] = None,
        advance_scene_rollout_period: Optional[int] = None,
        save_interval: Optional[int] = None,
        metric_accumulate_interval: Optional[int] = None,
    ):
        self._update_repeats: Optional[int] = None

        # Populate TrainingSettings members
        super().__init__(**prepare_locals_for_super(locals()))

        self.loss_names = loss_names
        self.max_stage_steps = max_stage_steps

        self.loss_weights = loss_weights
        self.loss_update_repeats = loss_update_repeats

        assert self.loss_weights is None or len(self.loss_weights) == len(
            self.loss_names
        )
        assert self.loss_update_repeats is None or (
            len(self.loss_update_repeats) == len(self.loss_names)
            and self._update_repeats is None
        )

        self.teacher_forcing = teacher_forcing
        self.offpolicy_component = offpolicy_component
        self.custom_components: List[
            CustomPipelineComponent
        ] = custom_components if custom_components is not None else []
        self.name_to_custom_component: Dict[str, CustomPipelineComponent] = {
            cc.uuid: cc for cc in self.custom_components
        }
        self.name_to_num_epoches: Dict[str, CustomPipelineComponent] = {
            cc.uuid: 0 for cc in self.custom_components
        }
        self.named_custom_storage: Optional[Dict[str, ExperienceStorage]] = None

        assert "onpolicy" not in self.name_to_custom_component
        assert "offpolicy" not in self.name_to_custom_component

        self.early_stopping_criterion = early_stopping_criterion

        self.steps_taken_in_stage: int = 0
        self.rollout_count = 0
        self.early_stopping_criterion_met = False

        self._component_name_to_named_loss_weights: Dict[str, Dict[str, float]] = {}
        self.component_name_to_stream_memory: Dict[str, Memory] = {
            cc.uuid: Memory() for cc in self.custom_components
        }
        self.component_name_to_num_epoches: Dict[str, int] = {
            cc.uuid: 0 for cc in self.custom_components
        }

        self.named_losses: Optional[Dict[str, AbstractActorCriticLoss]] = None
        self._named_loss_update_repeats: Optional[Dict[str, float]] = None

        self.offpolicy_memory = Memory()
        self.offpolicy_epochs: Optional[int] = None
        self.offpolicy_named_losses: Optional[Dict[str, AbstractOffPolicyLoss]] = None
        self.offpolicy_steps_taken_in_stage: int = 0

    @property
    def update_repeats(self) -> Optional[int]:
        if self._update_repeats is None:
            if self.loss_update_repeats is None:
                return None
            return max(self.loss_update_repeats)
        else:
            return self._update_repeats

    @update_repeats.setter
    def update_repeats(self, val: Optional[int]):
        self._update_repeats = val

    @property
    def is_complete(self):
        return (
            self.early_stopping_criterion_met
            or self.steps_taken_in_stage >= self.max_stage_steps
        )

    @property
    def named_loss_update_repeats(self):
        if self._named_loss_update_repeats is None:
            loss_update_repeats = (
                self.loss_update_repeats
                if self.loss_update_repeats is not None
                else [None] * len(self.loss_names)
            )
            self._named_loss_update_repeats = {
                name: weight
                for name, weight in zip(self.loss_names, loss_update_repeats)
            }
        return self._named_loss_update_repeats

    @property
    def named_loss_weights(self):
        return self.component_name_to_named_loss_weights("onpolicy")

    @property
    def offpolicy_named_loss_weights(self):
        return self.component_name_to_named_loss_weights("offpolicy")

    def component_name_to_named_loss_weights(self, uuid: str):
        if uuid not in self._component_name_to_named_loss_weights:
            if uuid == "onpolicy":
                loss_names = self.loss_names
                loss_weights = self.loss_weights
            elif uuid == "offpolicy":
                loss_names = self.offpolicy_component.loss_names
                loss_weights = self.offpolicy_component.loss_weights
            else:
                loss_names = self.name_to_custom_component[uuid].loss_names
                loss_weights = self.name_to_custom_component[uuid].loss_weights

            if loss_weights is None:
                loss_weights = [1.0] * len(loss_names)

            self._component_name_to_named_loss_weights[uuid] = {
                name: weight for name, weight in zip(loss_names, loss_weights)
            }

        return self._component_name_to_named_loss_weights[uuid]


class TrainingPipeline(TrainingSettings):
    """Class defining the stages (and global training settings) in a training
    pipeline.

    The training pipeline can be used as an iterator to go through the pipeline
    stages in, for instance, a loop.

    # Attributes

    named_losses : Dictionary mapping a the name of a loss to either an instantiation
        of that loss or a `Builder` that, when called, will return that loss.
    pipeline_stages : A list of PipelineStages. Each of these define how the agent
        will be trained and are executed sequentially.
    optimizer_builder : Builder object to instantiate the optimizer to use during training.
    num_mini_batch : See docs for `TrainingSettings`.
    update_repeats : See docs for `TrainingSettings`.
    max_grad_norm : See docs for `TrainingSettings`.
    num_steps : See docs for `TrainingSettings`.
    gamma : See docs for `TrainingSettings`.
    use_gae : See docs for `TrainingSettings`.
    gae_lambda : See docs for `TrainingSettings`.
    advance_scene_rollout_period: See docs for `TrainingSettings`.
    save_interval : See docs for `TrainingSettings`.
    metric_accumulate_interval : See docs for `TrainingSettings`.
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
        update_repeats: Optional[int],
        max_grad_norm: float,
        num_steps: int,
        gamma: float,
        use_gae: bool,
        gae_lambda: float,
        advance_scene_rollout_period: Optional[int],
        save_interval: Optional[int],
        metric_accumulate_interval: int,
        named_storages: Optional[
            Dict[str, Union[ExperienceStorage, Builder[ExperienceStorage]]]
        ] = None,
        should_log: bool = True,
        lr_scheduler_builder: Optional[Builder[optim.lr_scheduler._LRScheduler]] = None,  # type: ignore
    ):
        """Initializer.

        See class docstring for parameter definitions.
        """
        all_vars = prepare_locals_for_super(locals())

        # Populate TrainingSettings members
        super().__init__(**all_vars)

        self.optimizer_builder = optimizer_builder
        self.lr_scheduler_builder = lr_scheduler_builder

        self.named_losses = named_losses
        self.named_custom_storage = {} if named_storages is None else named_storages
        self.should_log = should_log

        self.pipeline_stages = pipeline_stages
        if len(self.pipeline_stages) > len(set(id(ps) for ps in pipeline_stages)):
            raise RuntimeError(
                "Duplicate `PipelineStage` object instances found in the pipeline stages input"
                " to `TrainingPipeline`. `PipelineStage` objects are not immutable, if you'd"
                " like to have multiple pipeline stages of the same type, please instantiate"
                " multiple separate instances."
            )

        self._current_stage: Optional[PipelineStage] = None
        for sit, stage in enumerate(self.pipeline_stages):
            # Forward all global `TrainingSettings` to all `PipelineStage`s unless overridden:
            for var in _TRAINING_SETTINGS_NAMES:
                if getattr(stage, var) is None:
                    setattr(stage, var, getattr(self, var))

            assert (
                stage.num_steps <= self.num_steps
            ), f"Stage {sit} has `num_steps` {stage.num_steps} > {self.num_steps} in pipeline."

        self.rollout_count = 0
        self.off_policy_epochs = None

        self._refresh_current_stage(force_stage_search_from_start=True)

    @property
    def total_steps(self) -> int:
        return sum(ps.steps_taken_in_stage for ps in self.pipeline_stages)

    @property
    def total_offpolicy_steps(self) -> int:
        return sum(ps.offpolicy_steps_taken_in_stage for ps in self.pipeline_stages)

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

    def before_rollout(self, train_metrics: Optional[ScalarMeanTracker] = None) -> bool:
        if (
            train_metrics is not None
            and self.current_stage.early_stopping_criterion is not None
        ):
            self.current_stage.early_stopping_criterion_met = self.current_stage.early_stopping_criterion(
                stage_steps=self.current_stage.steps_taken_in_stage,
                total_steps=self.total_steps,
                training_metrics=train_metrics,
            )
        if self.current_stage.early_stopping_criterion_met:
            get_logger().debug(
                f"Early stopping criterion met after {self.total_steps} total steps "
                f"({self.current_stage.steps_taken_in_stage} in current stage, stage index {self.current_stage_index})."
            )
        return self.current_stage is not self._refresh_current_stage(
            force_stage_search_from_start=False
        )

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
                    "offpolicy_steps_taken_in_stage": ps.offpolicy_steps_taken_in_stage,
                }
                for ps in self.pipeline_stages
            ],
            rollout_count=self.rollout_count,
            off_policy_epochs=self.off_policy_epochs,
        )

    def load_state_dict(self, state_dict: Dict[str, Any]):
        for ps, stage_info in zip(self.pipeline_stages, state_dict["stage_info_list"]):
            ps.early_stopping_criterion_met = stage_info["early_stopping_criterion_met"]
            ps.steps_taken_in_stage = stage_info["steps_taken_in_stage"]
            ps.offpolicy_steps_taken_in_stage = stage_info.get(
                "offpolicy_steps_taken_in_stage", 0
            )

        self.rollout_count = state_dict["rollout_count"]
        self.off_policy_epochs = state_dict.get("off_policy_epochs", 0)

        self._refresh_current_stage(force_stage_search_from_start=True)

    @property
    def current_custom_stage_storage(self) -> Dict[str, ExperienceStorage]:
        if self.current_stage.named_custom_storage is None:
            storage_names_for_current_stage = sorted(
                list(
                    set(cc.storage_name for cc in self.current_stage.custom_components)
                )
            )
            for storage_name in storage_names_for_current_stage:
                if isinstance(self.named_custom_storage[storage_name], Builder):
                    self.named_custom_storage[storage_name] = cast(
                        Builder["ExperienceStorage"],
                        self.named_custom_storage[storage_name],
                    )()

            self.current_stage.named_custom_storage = {
                storage_name: cast(
                    ExperienceStorage, self.named_custom_storage[storage_name]
                )
                for storage_name in storage_names_for_current_stage
            }

        return self.current_stage.named_custom_storage

    @property
    def current_stage_losses(self) -> Dict[str, AbstractActorCriticLoss]:
        if self.current_stage.named_losses is None:
            for loss_name in self.current_stage.loss_names:
                if isinstance(self.named_losses[loss_name], Builder):
                    self.named_losses[loss_name] = cast(
                        Builder["AbstractActorCriticLoss"],
                        self.named_losses[loss_name],
                    )()

            self.current_stage.named_losses = {
                loss_name: cast(AbstractActorCriticLoss, self.named_losses[loss_name])
                for loss_name in self.current_stage.loss_names
            }

        return self.current_stage.named_losses

    @property
    def current_stage_offpolicy_losses(self) -> Dict[str, AbstractOffPolicyLoss]:
        if self.current_stage.offpolicy_named_losses is None:
            for loss_name in self.current_stage.offpolicy_component.loss_names:
                if isinstance(self.named_losses[loss_name], Builder):
                    self.named_losses[loss_name] = cast(
                        Builder["AbstractOffPolicyLoss"], self.named_losses[loss_name],
                    )()

            self.current_stage.offpolicy_named_losses = {
                loss_name: cast(AbstractOffPolicyLoss, self.named_losses[loss_name])
                for loss_name in self.current_stage.offpolicy_component.loss_names
            }

        return self.current_stage.offpolicy_named_losses
