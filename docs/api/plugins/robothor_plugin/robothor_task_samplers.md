# plugins.robothor_plugin.robothor_task_samplers [[source]](https://github.com/allenai/embodied-rl/tree/master/plugins/robothor_plugin/robothor_task_samplers.py)

## ObjectNavDatasetTaskSampler
```python
ObjectNavDatasetTaskSampler(
    self,
    scenes: List[str],
    scene_directory: str,
    sensors: List[core.base_abstractions.sensor.Sensor],
    max_steps: int,
    env_args: Dict[str, Any],
    action_space: gym.spaces.space.Space,
    rewards_config: Dict,
    seed: Optional[int] = None,
    deterministic_cudnn: bool = False,
    loop_dataset: bool = True,
    allow_flipping = False,
    env_class = <class 'plugins.robothor_plugin.robothor_environment.RoboThorEnvironment'>,
    args,
    kwargs,
) -> None
```

### all_observation_spaces_equal
Check if observation spaces equal.

__Returns__


True if all Tasks that can be sampled by this sampler have the
    same observation space. Otherwise False.

### length
Length.

__Returns__


Number of total tasks remaining that can be sampled. Can be float('inf').

## ObjectNavTaskSampler
```python
ObjectNavTaskSampler(
    self,
    scenes: Union[List[str], str],
    object_types: List[str],
    sensors: List[core.base_abstractions.sensor.Sensor],
    max_steps: int,
    env_args: Dict[str, Any],
    action_space: gym.spaces.space.Space,
    rewards_config: Dict,
    scene_period: Optional[int, str] = None,
    max_tasks: Optional[int] = None,
    seed: Optional[int] = None,
    deterministic_cudnn: bool = False,
    allow_flipping: bool = False,
    dataset_first: int = -1,
    dataset_last: int = -1,
    args,
    kwargs,
) -> None
```

### all_observation_spaces_equal
Check if observation spaces equal.

__Returns__


True if all Tasks that can be sampled by this sampler have the
same observation space. Otherwise False.

### length
Length.

__Returns__


Number of total tasks remaining that can be sampled. Can be float('inf').

## PointNavDatasetTaskSampler
```python
PointNavDatasetTaskSampler(
    self,
    scenes: List[str],
    scene_directory: str,
    sensors: List[core.base_abstractions.sensor.Sensor],
    max_steps: int,
    env_args: Dict[str, Any],
    action_space: gym.spaces.space.Space,
    rewards_config: Dict,
    seed: Optional[int] = None,
    deterministic_cudnn: bool = False,
    loop_dataset: bool = True,
    shuffle_dataset: bool = True,
    allow_flipping = False,
    env_class = <class 'plugins.robothor_plugin.robothor_environment.RoboThorEnvironment'>,
    args,
    kwargs,
) -> None
```

### all_observation_spaces_equal
Check if observation spaces equal.

__Returns__


True if all Tasks that can be sampled by this sampler have the
    same observation space. Otherwise False.

### length
Length.

__Returns__


Number of total tasks remaining that can be sampled.
Can be float('inf').

## PointNavTaskSampler
```python
PointNavTaskSampler(
    self,
    scenes: List[str],
    sensors: List[core.base_abstractions.sensor.Sensor],
    max_steps: int,
    env_args: Dict[str, Any],
    action_space: gym.spaces.space.Space,
    rewards_config: Dict,
    scene_period: Optional[int, str] = None,
    max_tasks: Optional[int] = None,
    seed: Optional[int] = None,
    deterministic_cudnn: bool = False,
    fixed_tasks: Optional[List[Dict[str, Any]]] = None,
    args,
    kwargs,
) -> None
```

### all_observation_spaces_equal
Check if observation spaces equal.

__Returns__


True if all Tasks that can be sampled by this sampler
have the     same observation space. Otherwise False.

### length
Length.

__Returns__


Number of total tasks remaining that can be sampled.
Can be float('inf').

