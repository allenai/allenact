# core.algorithms.onpolicy_sync.vector_sampled_tasks [[source]](https://github.com/allenai/allenact/tree/master/core/algorithms/onpolicy_sync/vector_sampled_tasks.py)

## SingleProcessVectorSampledTasks
```python
SingleProcessVectorSampledTasks(
    self,
    make_sampler_fn: Callable[..., core.base_abstractions.task.TaskSampler],
    sampler_fn_args_list: Sequence[Dict[str, Any]] = None,
    auto_resample_when_done: bool = True,
    should_log: bool = True,
    metrics_out_queue: Optional[queue.Queue] = None,
) -> None
```
Vectorized collection of tasks.

Simultaneously handles the state of multiple TaskSamplers and their associated tasks.
Allows for interacting with these tasks in a vectorized manner. When a task completes,
another task is sampled from the appropriate task sampler. All the tasks are
synchronized (for step and new_task methods).

__Attributes__


- `make_sampler_fn `: function which creates a single TaskSampler.
- `sampler_fn_args `: sequence of dictionaries describing the args
    to pass to make_sampler_fn on each individual process.
- `auto_resample_when_done `: automatically sample a new Task from the TaskSampler when
    the Task completes. If False, a new Task will not be resampled until all
    Tasks on all processes have completed. This functionality is provided for seamless training
    of vectorized Tasks.

### attr
```python
SingleProcessVectorSampledTasks.attr(
    self,
    attr_names: Union[List[str], str],
) -> List[Any]
```
Gets the attributes (specified by name) on the tasks.

__Parameters__


- __attr_names __: The name of the functions to call on the tasks.

__Returns__


List of results of calling the functions.

### attr_at
```python
SingleProcessVectorSampledTasks.attr_at(
    self,
    sampler_index: int,
    attr_name: str,
) -> Any
```
Gets the attribute (specified by name) on the selected task and
returns it.

__Parameters__


- __index __: Which task to call the function on.
- __attr_name __: The name of the function to call on the task.

__Returns__


 Result of calling the function.

### call
```python
SingleProcessVectorSampledTasks.call(
    self,
    function_names: Union[str, List[str]],
    function_args_list: Optional[List[Any]] = None,
) -> List[Any]
```
Calls a list of functions (which are passed by name) on the
corresponding task (by index).

__Parameters__


- __function_names __: The name of the functions to call on the tasks.
- __function_args_list __: List of function args for each function.
    If provided, len(function_args_list) should be as long as  len(function_names).

__Returns__


List of results of calling the functions.

### call_at
```python
SingleProcessVectorSampledTasks.call_at(
    self,
    sampler_index: int,
    function_name: str,
    function_args: Optional[List[Any]] = None,
) -> Any
```
Calls a function (which is passed by name) on the selected task and
returns the result.

__Parameters__


- __index __: Which task to call the function on.
- __function_name __: The name of the function to call on the task.
- __function_args __: Optional function args.

__Returns__


Result of calling the function.

### command_at
```python
SingleProcessVectorSampledTasks.command_at(
    self,
    sampler_index: int,
    command: str,
    data: Optional[Any] = None,
) -> Any
```
Calls a function (which is passed by name) on the selected task and
returns the result.

__Parameters__


- __index __: Which task to call the function on.
- __function_name __: The name of the function to call on the task.
- __function_args __: Optional function args.

__Returns__


Result of calling the function.

### get_observations
```python
SingleProcessVectorSampledTasks.get_observations(self)
```
Get observations for all unpaused tasks.

__Returns__


List of observations for each of the unpaused tasks.

### is_closed
Has the vector task been closed.
### next_task
```python
SingleProcessVectorSampledTasks.next_task(self, **kwargs)
```
Move to the the next Task for all TaskSamplers.

__Parameters__


- __kwargs __: key word arguments passed to the `next_task` function of the samplers.

__Returns__


List of initial observations for each of the new tasks.

### next_task_at
```python
SingleProcessVectorSampledTasks.next_task_at(
    self,
    index_process: int,
) -> List[core.base_abstractions.misc.RLStepResult]
```
Move to the the next Task from the TaskSampler in index_process
process in the vector.

__Parameters__


- __index_process __: Index of the generator to be reset.

__Returns__


List of length one containing the observations the newly sampled task.

### num_unpaused_tasks
Number of unpaused processes.

__Returns__


Number of unpaused processes.

### pause_at
```python
SingleProcessVectorSampledTasks.pause_at(self, sampler_index: int) -> None
```
Pauses computation on the Task in process `index` without destroying
the Task. This is useful for not needing to call steps on all Tasks
when only some are active (for example during the last samples of
running eval).

__Parameters__


- __index __: which process to pause. All indexes after this
    one will be shifted down by one.

### render
```python
SingleProcessVectorSampledTasks.render(
    self,
    mode: str = 'human',
    args,
    kwargs,
) -> Union[numpy.ndarray, NoneType, List[numpy.ndarray]]
```
Render observations from all Tasks in a tiled image or a list of
images.
### reset_all
```python
SingleProcessVectorSampledTasks.reset_all(self)
```
Reset all task samplers to their initial state (except for the RNG
seed).
### resume_all
```python
SingleProcessVectorSampledTasks.resume_all(self) -> None
```
Resumes any paused processes.
### set_seeds
```python
SingleProcessVectorSampledTasks.set_seeds(self, seeds: List[int])
```
Sets new tasks' RNG seeds.

__Parameters__


- __seeds__: List of size _num_samplers containing new RNG seeds.

### step
```python
SingleProcessVectorSampledTasks.step(self, actions: List[int])
```
Perform actions in the vectorized tasks.

__Parameters__


- __actions__: List of size _num_samplers containing action to be taken in each task.

__Returns__


List of outputs from the step method of tasks.

### step_at
```python
SingleProcessVectorSampledTasks.step_at(
    self,
    index_process: int,
    action: int,
) -> List[core.base_abstractions.misc.RLStepResult]
```
Step in the index_process task in the vector.

__Parameters__


- __index_process __: Index of the process to be reset.
- __action __: The action to take.

__Returns__


List containing the output of step method on the task in the indexed process.

## VectorSampledTasks
```python
VectorSampledTasks(
    self,
    make_sampler_fn: Callable[..., core.base_abstractions.task.TaskSampler],
    sampler_fn_args: Sequence[Dict[str, Any]] = None,
    auto_resample_when_done: bool = True,
    multiprocessing_start_method: Optional[str] = 'forkserver',
    mp_ctx: Optional[multiprocessing.context.BaseContext] = None,
    metrics_out_queue: <bound method BaseContext.Queue of <multiprocessing.context.DefaultContext object at 0x10f18b250>> = None,
    should_log: bool = True,
    max_processes: Optional[int] = None,
) -> None
```
Vectorized collection of tasks. Creates multiple processes where each
process runs its own TaskSampler. Each process generates one Task from its
TaskSampler at a time and this class allows for interacting with these
tasks in a vectorized manner. When a task on a process completes, the
process samples another task from its task sampler. All the tasks are
synchronized (for step and new_task methods).

__Attributes__


- `make_sampler_fn `: function which creates a single TaskSampler.
- `sampler_fn_args `: sequence of dictionaries describing the args
    to pass to make_sampler_fn on each individual process.
- `auto_resample_when_done `: automatically sample a new Task from the TaskSampler when
    the Task completes. If False, a new Task will not be resampled until all
    Tasks on all processes have completed. This functionality is provided for seamless training
    of vectorized Tasks.
- `multiprocessing_start_method `: the multiprocessing method used to
    spawn worker processes. Valid methods are
    ``{'spawn', 'forkserver', 'fork'}`` ``'forkserver'`` is the
    recommended method as it works well with CUDA. If
    ``'fork'`` is used, the subproccess  must be started before
    any other GPU useage.

### async_step
```python
VectorSampledTasks.async_step(self, actions: List[int]) -> None
```
Asynchronously step in the vectorized Tasks.

__Parameters__


- __actions __: actions to be performed in the vectorized Tasks.

### attr
```python
VectorSampledTasks.attr(self, attr_names: Union[List[str], str]) -> List[Any]
```
Gets the attributes (specified by name) on the tasks.

__Parameters__


- __attr_names __: The name of the functions to call on the tasks.

__Returns__


List of results of calling the functions.

### attr_at
```python
VectorSampledTasks.attr_at(self, sampler_index: int, attr_name: str) -> Any
```
Gets the attribute (specified by name) on the selected task and
returns it.

__Parameters__


- __index __: Which task to call the function on.
- __attr_name __: The name of the function to call on the task.

__Returns__


 Result of calling the function.

### call
```python
VectorSampledTasks.call(
    self,
    function_names: Union[str, List[str]],
    function_args_list: Optional[List[Any]] = None,
) -> List[Any]
```
Calls a list of functions (which are passed by name) on the
corresponding task (by index).

__Parameters__


- __function_names __: The name of the functions to call on the tasks.
- __function_args_list __: List of function args for each function.
    If provided, len(function_args_list) should be as long as  len(function_names).

__Returns__


List of results of calling the functions.

### call_at
```python
VectorSampledTasks.call_at(
    self,
    sampler_index: int,
    function_name: str,
    function_args: Optional[List[Any]] = None,
) -> Any
```
Calls a function (which is passed by name) on the selected task and
returns the result.

__Parameters__


- __index __: Which task to call the function on.
- __function_name __: The name of the function to call on the task.
- __function_args __: Optional function args.

__Returns__


Result of calling the function.

### command_at
```python
VectorSampledTasks.command_at(
    self,
    sampler_index: int,
    command: str,
    data: Optional[Any] = None,
) -> Any
```
Runs the command on the selected task and returns the result.

__Parameters__



__Returns__


Result of the command.

### get_observations
```python
VectorSampledTasks.get_observations(self)
```
Get observations for all unpaused tasks.

__Returns__


List of observations for each of the unpaused tasks.

### is_closed
Has the vector task been closed.
### mp_ctx
Get the multiprocessing process used by the vector task.

__Returns__


The multiprocessing context.

### next_task
```python
VectorSampledTasks.next_task(self, **kwargs)
```
Move to the the next Task for all TaskSamplers.

__Parameters__


- __kwargs __: key word arguments passed to the `next_task` function of the samplers.

__Returns__


List of initial observations for each of the new tasks.

### next_task_at
```python
VectorSampledTasks.next_task_at(
    self,
    sampler_index: int,
) -> List[core.base_abstractions.misc.RLStepResult]
```
Move to the the next Task from the TaskSampler in index_process
process in the vector.

__Parameters__


- __index_process __: Index of the process to be reset.

__Returns__


List of length one containing the observations the newly sampled task.

### num_unpaused_tasks
Number of unpaused processes.

__Returns__


Number of unpaused processes.

### pause_at
```python
VectorSampledTasks.pause_at(self, sampler_index: int) -> None
```
Pauses computation on the Task in process `index` without destroying
the Task. This is useful for not needing to call steps on all Tasks
when only some are active (for example during the last samples of
running eval).

__Parameters__


- __index __: which process to pause. All indexes after this
    one will be shifted down by one.

### render
```python
VectorSampledTasks.render(
    self,
    mode: str = 'human',
    args,
    kwargs,
) -> Union[numpy.ndarray, NoneType, List[numpy.ndarray]]
```
Render observations from all Tasks in a tiled image or list of
images.
### reset_all
```python
VectorSampledTasks.reset_all(self)
```
Reset all task samplers to their initial state (except for the RNG
seed).
### resume_all
```python
VectorSampledTasks.resume_all(self) -> None
```
Resumes any paused processes.
### set_seeds
```python
VectorSampledTasks.set_seeds(self, seeds: List[int])
```
Sets new tasks' RNG seeds.

__Parameters__


- __seeds__: List of size _num_samplers containing new RNG seeds.

### step
```python
VectorSampledTasks.step(self, actions: List[int])
```
Perform actions in the vectorized tasks.

__Parameters__


- __actions__: List of size _num_samplers containing action to be taken in each task.

__Returns__


List of outputs from the step method of tasks.

### step_at
```python
VectorSampledTasks.step_at(
    self,
    sampler_index: int,
    action: int,
) -> List[core.base_abstractions.misc.RLStepResult]
```
Step in the index_process task in the vector.

__Parameters__


- __sampler_index __: Index of the sampler to be reset.
- __action __: The action to take.

__Returns__


List containing the output of step method on the task in the indexed process.

### wait_step
```python
VectorSampledTasks.wait_step(self) -> List[Dict[str, Any]]
```
Wait until all the asynchronized processes have synchronized.
