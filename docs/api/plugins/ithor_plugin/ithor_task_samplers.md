# plugins.ithor_plugin.ithor_task_samplers [[source]](https://github.com/allenai/allenact/tree/master/plugins/ithor_plugin/ithor_task_samplers.py)

## ObjectNavTaskSampler
```python
ObjectNavTaskSampler(
    self,
    scenes: List[str],
    object_types: str,
    sensors: List[core.base_abstractions.sensor.Sensor],
    max_steps: int,
    env_args: Dict[str, Any],
    action_space: gym.spaces.space.Space,
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


True if all Tasks that can be sampled by this sampler have the
    same observation space. Otherwise False.

### length
Length.

__Returns__


Number of total tasks remaining that can be sampled. Can be float('inf').

