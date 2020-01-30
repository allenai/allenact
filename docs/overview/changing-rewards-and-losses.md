# Changing rewards and losses

In order to train actor-critic agents, we need to specify

* `rewards` at the task level
* `losses` at the training pipeline level 

## Rewards

For example, taking an [object navigation task in AI2THOR](/api/extensions/ai2thor/tasks/#objectnavtask) as a starting
 point, we can see how the `_step(self, action: int) -> RLStepResult` method computes the reward for the latest action by invoking a function like:

```python
def judge(self) -> float:
    reward = -0.01

    if not self.last_action_success:
        reward += -0.1

    if self._took_end_action:
        reward += 1.0 if self._success else -1.0

    return float(reward)
```

Any reward shaping can be easily added by e.g. modifying the definition of an existing class:

```python
class NavigationWithShaping(rl_ai2thor.tasks.ObjectNavTask):
    def judge(self) -> float:
        reward = -0.01
        
        #TODO

``` 

## Losses