# plugins.minigrid_plugin.minigrid_environments [[source]](https://github.com/allenai/embodied-rl/tree/master/plugins/minigrid_plugin/minigrid_environments.py)

## AskForHelpSimpleCrossing
```python
AskForHelpSimpleCrossing(
    self,
    size = 9,
    num_crossings = 1,
    obstacle_type = <class 'gym_minigrid.minigrid.Wall'>,
    seed = None,
    exploration_reward: Optional[float] = None,
    death_penalty: Optional[float] = None,
    toggle_is_permenant: bool = False,
)
```
Corresponds to WC FAULTY SWITCH environment.
### step
```python
AskForHelpSimpleCrossing.step(self, action: int)
```
Reveal the observation only if the `toggle` action is executed.
## FastCrossing
```python
FastCrossing(
    self,
    size = 9,
    num_crossings = 1,
    obstacle_type = <class 'gym_minigrid.minigrid.Lava'>,
    seed = None,
)
```
Similar to `CrossingEnv`, but to support faster task sampling as per
`repeat_failed_task_for_min_steps` flag in MiniGridTaskSampler.
