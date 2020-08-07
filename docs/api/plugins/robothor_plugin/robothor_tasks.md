# plugins.robothor_plugin.robothor_tasks [[source]](https://github.com/allenai/embodied-rl/tree/master/plugins/robothor_plugin/robothor_tasks.py)

## ObjectNavTask
```python
ObjectNavTask(
    self,
    env: plugins.robothor_plugin.robothor_environment.RoboThorEnvironment,
    sensors: List[core.base_abstractions.sensor.Sensor],
    task_info: Dict[str, Any],
    max_steps: int,
    reward_configs: Dict[str, Any],
    distance_cache: Optional[Dict[str, Any]] = None,
    kwargs,
) -> None
```

### judge
```python
ObjectNavTask.judge(self) -> float
```
Judge the last event.
## PointNavTask
```python
PointNavTask(
    self,
    env: plugins.robothor_plugin.robothor_environment.RoboThorEnvironment,
    sensors: List[core.base_abstractions.sensor.Sensor],
    task_info: Dict[str, Any],
    max_steps: int,
    reward_configs: Dict[str, Any],
    distance_cache: Optional[Dict[str, Any]] = None,
    episode_info: Optional[Dict[str, Any]] = None,
    kwargs,
) -> None
```

### judge
```python
PointNavTask.judge(self) -> float
```
Judge the last event.
