# core.base_abstractions.task [[source]](https://github.com/allenai/embodied-rl/tree/master/core/base_abstractions/task.py)
Defines the primary data structures by which agents interact with their
environment.
## EnvType
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

## SubTaskType
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

## Task
```python
Task(
    self,
    env: ~EnvType,
    sensors: Union[core.base_abstractions.sensor.SensorSuite, Sequence[core.base_abstractions.sensor.Sensor]],
    task_info: Dict[str, Any],
    max_steps: int,
    kwargs,
) -> None
```
An abstract class defining a, goal directed, 'task.' Agents interact
with their environment through a task by taking a `step` after which they
receive new observations, rewards, and (potentially) other useful
information.

A Task is a helpful generalization of the OpenAI gym's `Env` class
and allows for multiple tasks (e.g. point and object navigation) to
be defined on a single environment (e.g. AI2-THOR).

__Attributes__


- `env `: The environment.
- `sensor_suite`: Collection of sensors formed from the `sensors` argument in the initializer.
- `task_info `: Dictionary of (k, v) pairs defining task goals and other task information.
- `max_steps `: The maximum number of steps an agent can take an in the task before it is considered failed.
- `observation_space`: The observation space returned on each step from the sensors.

### action_names
```python
Task.action_names(self) -> Tuple[str, ...]
```
Action names of the Task instance.

This method should be overwritten if `class_action_names`
requires key word arguments to determine the number of actions.

### action_space
Task's action space.

__Returns__


The action space for the task.

### class_action_names
```python
Task.class_action_names(**kwargs) -> Tuple[str, ...]
```
A tuple of action names.

__Parameters__


- __kwargs __: Keyword arguments.

__Returns__


Tuple of (ordered) action names so that taking action
    running `task.step(i)` corresponds to taking action task.class_action_names()[i].

### close
```python
Task.close(self) -> None
```
Closes the environment and any other files opened by the Task (if
applicable).
### cumulative_reward
Total cumulative in the task so far.

__Returns__


Cumulative reward as a float.

### index_to_action
```python
Task.index_to_action(self, index: int) -> str
```
Returns the action name correspond to `index`.
### is_done
```python
Task.is_done(self) -> bool
```
Did the agent reach a terminal state or performed the maximum number
of steps.
### metrics
```python
Task.metrics(self) -> Dict[str, Any]
```
Computes metrics related to the task after the task's completion.

By default this function is automatically called during training
and the reported metrics logged to tensorboard.

__Returns__


A dictionary where every key is a string (the metric's
    name) and the value is the value of the metric.

### num_steps_taken
```python
Task.num_steps_taken(self) -> int
```
Number of steps taken by the agent in the task so far.
### query_expert
```python
Task.query_expert(self, **kwargs) -> Tuple[Any, bool]
```
Query the expert policy for this task.

__Returns__


 A tuple (x, y) where x is the expert action (or policy) and y is False             if the expert could not determine the optimal action (otherwise True). Here y             is used for masking. Even when y is False, x should still lie in the space of             possible values (e.g. if x is the expert policy then x should be the correct length,             sum to 1, and have non-negative entries).

### reached_max_steps
```python
Task.reached_max_steps(self) -> bool
```
Has the agent reached the maximum number of steps.
### reached_terminal_state
```python
Task.reached_terminal_state(self) -> bool
```
Has the agent reached a terminal state (excluding reaching the
maximum number of steps).
### render
```python
Task.render(self, mode: str = 'rgb', *args, **kwargs) -> numpy.ndarray
```
Render the current task state.

Rendered task state can come in any supported modes.

__Parameters__


- __mode __: The mode in which to render. For example, you might have a 'rgb'
    mode that renders the agent's egocentric viewpoint or a 'dev' mode
    returning additional information.
- __args __: Extra args.
- __kwargs __: Extra kwargs.

__Returns__


An numpy array corresponding to the requested render.

### step
```python
Task.step(self, action: int) -> core.base_abstractions.misc.RLStepResult
```
Take an action in the environment.

Takes the action in the environment corresponding to
`self.class_action_names()[action]` and returns
observations (& rewards and any additional information)
corresponding to the agent's new state. Note that this function
should not be overwritten without care (instead
implement the `_step` function).

__Parameters__


- __action __: The action to take.

__Returns__


A `RLStepResult` object encoding the new observations, reward, and
(possibly) additional information.

### total_actions
Total number of actions available to an agent in this Task.
## TaskSampler
```python
TaskSampler(self, /, *args, **kwargs)
```
Abstract class defining a how new tasks are sampled.
### all_observation_spaces_equal
Checks if all observation spaces of tasks that can be sampled are
equal.

This will almost always simply return `True`. A case in which it should
return `False` includes, for example, a setting where you design
a `TaskSampler` that can generate different types of tasks, i.e.
point navigation tasks and object navigation tasks. In this case, these
different tasks may output different types of observations.

__Returns__


True if all Tasks that can be sampled by this sampler have the
    same observation space. Otherwise False.

### close
```python
TaskSampler.close(self) -> None
```
Closes any open environments or streams.

Should be run when done sampling.

### last_sampled_task
Get the most recently sampled Task.

__Returns__


The most recently sampled Task.

### length
Length.

__Returns__


Number of total tasks remaining that can be sampled. Can be
    float('inf').

### next_task
```python
TaskSampler.next_task(
    self,
    force_advance_scene: bool = False,
) -> Optional[core.base_abstractions.task.Task]
```
Get the next task in the sampler's stream.

__Parameters__


- __force_advance_scene __: Used to (if applicable) force the task sampler to
    use a new scene for the next task. This is useful if, during training,
    you would like to train with one scene for some number of steps and
    then explicitly control when you begin training with the next scene.

__Returns__


The next Task in the sampler's stream if a next task exists. Otherwise None.

### reset
```python
TaskSampler.reset(self) -> None
```
Resets task sampler to its original state (except for any seed).
### set_seed
```python
TaskSampler.set_seed(self, seed: int) -> None
```
Sets new RNG seed.

__Parameters__


- __seed __: New seed.

### total_unique
Total unique tasks.

__Returns__


Total number of *unique* tasks that can be sampled. Can be
    float('inf') or, if the total unique is not known, None.

