# plugins.minigrid_plugin.minigrid_tasks [[source]](https://github.com/allenai/allenact/tree/master/plugins/minigrid_plugin/minigrid_tasks.py)

## MiniGridTask
```python
MiniGridTask(
    self,
    env: gym_minigrid.envs.crossing.CrossingEnv,
    sensors: Union[core.base_abstractions.sensor.SensorSuite, List[core.base_abstractions.sensor.Sensor]],
    task_info: Dict[str, Any],
    max_steps: int,
    task_cache_uid: Optional[str] = None,
    corrupt_expert_within_actions_of_goal: Optional[int] = None,
    kwargs,
)
```

### generate_graph
```python
MiniGridTask.generate_graph(self) -> networkx.classes.digraph.DiGraph
```
The generated graph is based on the fully observable grid (as the
expert sees it all).

env: environment to generate the graph over

