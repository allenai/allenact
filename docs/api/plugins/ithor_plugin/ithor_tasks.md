# plugins.ithor_plugin.ithor_tasks [[source]](https://github.com/allenai/embodied-rl/tree/master/plugins/ithor_plugin/ithor_tasks.py)

## ObjectNavTask
```python
ObjectNavTask(
    self,
    env: plugins.ithor_plugin.ithor_environment.IThorEnvironment,
    sensors: List[core.base_abstractions.sensor.Sensor],
    task_info: Dict[str, Any],
    max_steps: int,
    kwargs,
) -> None
```
Defines the object navigation task in AI2-THOR.

In object navigation an agent is randomly initialized into an AI2-THOR scene and must
find an object of a given type (e.g. tomato, television, etc). An object is considered
found if the agent takes an `End` action and the object is visible to the agent (see
[here](https://ai2thor.allenai.org/documentation/concepts) for a definition of visibiliy
in AI2-THOR).

The actions available to an agent in this task are:

1. Move ahead
    * Moves agent ahead by 0.25 meters.
1. Rotate left / rotate right
    * Rotates the agent by 90 degrees counter-clockwise / clockwise.
1. Look down / look up
    * Changes agent view angle by 30 degrees up or down. An agent cannot look more than 30
      degrees above horizontal or less than 60 degrees below horizontal.
1. End
    * Ends the task and the agent receives a positive reward if the object type is visible to the agent,
    otherwise it receives a negative reward.

__Attributes__


- `env `: The ai2thor environment.
- `sensor_suite`: Collection of sensors formed from the `sensors` argument in the initializer.
- `task_info `: The task info. Must contain a field "object_type" that specifies, as a string,
    the goal object type.
- `max_steps `: The maximum number of steps an agent can take an in the task before it is considered failed.
- `observation_space`: The observation space returned on each step from the sensors.

### judge
```python
ObjectNavTask.judge(self) -> float
```
Compute the reward after having taken a step.
