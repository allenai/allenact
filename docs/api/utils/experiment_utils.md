# utils.experiment_utils [[source]](https://github.com/allenai/embodied-rl/tree/master/utils/experiment_utils.py)
Utility classes and functions for running and designing experiments.
## Builder
```python
Builder(self, /, *args, **kwargs)
```
Used to instantiate a given class with (default) parameters.

Helper class that stores a class, default parameters for that
class, and key word arguments that (possibly) overwrite the defaults.
When calling this an object of the Builder class it generates
a class of type `class_type` with parameters specified by
the attributes `default` and `kwargs` (and possibly additional, overwriting,
keyword arguments).

__Attributes__


- `class_type `: The class to be instantiated when calling the object.
- `kwargs `: Keyword arguments used to instantiate an object of type `class_type`.
- `default `: Default parameters used when instantiating the class.

## EarlyStoppingCriterion
```python
EarlyStoppingCriterion(self, /, *args, **kwargs)
```
Abstract class for class who determines if training should stop early in
a particular pipeline stage.
## LinearDecay
```python
LinearDecay(self, steps: int, startp: float = 1.0, endp: float = 0.0) -> None
```
Linearly decay between two values over some number of steps.

Obtain the value corresponding to the `i`th step by calling
an instantiation of this object with the value `i`.

__Parameters__


- __steps __: The number of steps over which to decay.
- __startp __: The starting value.
- __endp __: The ending value.

## NeverEarlyStoppingCriterion
```python
NeverEarlyStoppingCriterion(self, /, *args, **kwargs)
```
Implementation of `EarlyStoppingCriterion` which never stops early.
## OffPolicyPipelineComponent
```python
OffPolicyPipelineComponent(self, /, *args, **kwargs)
```
OffPolicyPipelineComponent(data_iterator_builder, loss_names, updates, loss_weights)
### data_iterator_builder
Alias for field number 0
### loss_names
Alias for field number 1
### loss_weights
Alias for field number 3
### updates
Alias for field number 2
## PipelineStage
```python
PipelineStage(
    self,
    loss_names: List[str],
    max_stage_steps: Union[int, Callable],
    early_stopping_criterion: Optional[utils.experiment_utils.EarlyStoppingCriterion] = None,
    loss_weights: Optional[Sequence[float]] = None,
    teacher_forcing: Optional[utils.experiment_utils.LinearDecay] = None,
    offpolicy_component: Optional[utils.experiment_utils.OffPolicyPipelineComponent] = None,
)
```
A single stage in a training pipeline.

__Attributes__


- `loss_name `: A collection of unique names assigned to losses. These will
    reference the `Loss` objects in a `TrainingPipeline` instance.
- `max_stage_steps `: Either the total number of steps agents should take in this stage or
    a Callable object (e.g. a function)
- `early_stopping_criterion`: An `EarlyStoppingCriterion` object which determines if
    training in this stage should be stopped early. If `None` then no early stopping
    occurs. If `early_stopping_criterion` is not `None` then we do not guarantee
    reproducibility when restarting a model from a checkpoint (as the
     `EarlyStoppingCriterion` object may store internal state which is not
     saved in the checkpoint).
- `loss_weights `: A list of floating point numbers describing the relative weights
    applied to the losses referenced by `loss_name`. Should be the same length
    as `loss_name`. If this is `None`, all weights will be assumed to be one.
- `teacher_forcing `: If applicable, defines the probability an agent will take the
    expert action (as opposed to its own sampled action) at a given time point.

## recursive_update
```python
recursive_update(
    original: Union[Dict, collections.abc.MutableMapping],
    update: Union[Dict, collections.abc.MutableMapping],
)
```
Recursively updates original dictionary with entries form update dict.

__Parameters__


- __original __: Original dictionary to be updated.
- __update __: Dictionary with additional or replacement entries.

__Returns__


Updated original dictionary.

## ScalarMeanTracker
```python
ScalarMeanTracker(self) -> None
```
Track a collection `scalar key -> mean` pairs.
### add_scalars
```python
ScalarMeanTracker.add_scalars(
    self,
    scalars: Dict[str, Union[float, int]],
    n: int = 1,
) -> None
```
Add additional scalars to track.

__Parameters__


- __scalars __: A dictionary of `scalar key -> value` pairs.

### pop_and_reset
```python
ScalarMeanTracker.pop_and_reset(self) -> Dict[str, float]
```
Return tracked means and reset.

On resetting all previously tracked values are discarded.

__Returns__


A dictionary of `scalar key -> current mean` pairs corresponding to those
values added with `add_scalars`.

## set_deterministic_cudnn
```python
set_deterministic_cudnn() -> None
```
Makes cudnn deterministic.

This may slow down computations.

## set_seed
```python
set_seed(seed: Union[int, NoneType] = None) -> None
```
Set seeds for multiple (cpu) sources of randomness.

Sets seeds for (cpu) `pytorch`, base `random`, and `numpy`.

__Parameters__


- __seed __: The seed to set. If set to None, keep using the current seed.

## ToBuildType
Type variable.

Usage::

  T = TypeVar('T')  # Can be anything
  A = TypeVar('A', str, bytes)  # Must be str or bytes

Type variables exist primarily for the benefit of static type
checkers.  They serve as the parameters for generic types as well
as for generic function definitions.  See class Generic for more
information on generic types.  Generic functions work as follows:

  def repeat(x: T, n: int) -> List[T]:
      '''Return a list containing n references to x.'''
      return [x]*n

  def longest(x: A, y: A) -> A:
      '''Return the longest of two strings.'''
      return x if len(x) >= len(y) else y

The latter example's signature is essentially the overloading
of (str, str) -> str and (bytes, bytes) -> bytes.  Also note
that if the arguments are instances of some subclass of str,
the return type is still plain str.

At runtime, isinstance(x, T) and issubclass(C, T) will raise TypeError.

Type variables defined with covariant=True or contravariant=True
can be used to declare covariant or contravariant generic types.
See PEP 484 for more details. By default generic types are invariant
in all type variables.

Type variables can be introspected. e.g.:

  T.__name__ == 'T'
  T.__constraints__ == ()
  T.__covariant__ == False
  T.__contravariant__ = False
  A.__constraints__ == (str, bytes)

Note that only type variables defined in global scope can be pickled.

## TrainingPipeline
```python
TrainingPipeline(
    self,
    named_losses: Dict[str, Union[core.base_abstractions.misc.Loss, utils.experiment_utils.Builder[core.base_abstractions.misc.Loss]]],
    pipeline_stages: List[utils.experiment_utils.PipelineStage],
    optimizer_builder: utils.experiment_utils.Builder[torch.optim.optimizer.Optimizer],
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
    lr_scheduler_builder: Optional[utils.experiment_utils.Builder[torch.optim.lr_scheduler._LRScheduler]] = None,
)
```
Class defining the stages (and global parameters) in a training
pipeline.

The training pipeline can be used as an iterator to go through the pipeline
stages in, for instance, a loop.

__Attributes__


- `named_losses `: Dictionary mapping a the name of a loss to either an instantiation
    of that loss or a `Builder` that, when called, will return that loss.
- `pipeline_stages `: A list of PipelineStages. Each of these define how the agent
    will be trained and are executed sequentially.
- `optimizer_builder `: Builder object to instantiate the optimizer to use during training.
- `num_mini_batch `: The number of mini-batches to break a rollout into.
- `update_repeats `: The number of times we will cycle through the mini-batches corresponding
    to a single rollout doing gradient updates.
- `max_grad_norm `: The maximum "inf" norm of any gradient step (gradients are clipped to not exceed this).
- `num_steps `: Total number of steps a single agent takes in a rollout.
- `gamma `: Discount factor applied to rewards (should be in [0, 1]).
- `use_gae `: Whether or not to use generalized advantage estimation (GAE).
- `gae_lambda `: The additional parameter used in GAE.
- `save_interval `: The frequency with which to save (in total agent steps taken). If `None` then *no*
    checkpoints will be saved. Otherwise, in addition to the checkpoints being saved every
    `save_interval` steps, a checkpoint will *always* be saved at the end of each pipeline stage.
    If `save_interval <= 0` then checkpoints will only be saved at the end of each pipeline stage.
- `metric_accumulate_interval `: The frequency with which training/validation metrics are accumulated
    (in total agent steps). Metrics accumulated in an interval are logged (if `should_log` is `True`)
    and used by the stage's early stopping criterion (if any).
- `should_log`: `True` if metrics accumulated during training should be logged to the console as well
    as to a tensorboard file.
- `lr_scheduler_builder `: Optional builder object to instantiate the learning rate scheduler used
    through the pipeline.

