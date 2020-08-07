# core.algorithms.onpolicy_sync.losses.ppo [[source]](https://github.com/allenai/embodied-rl/tree/master/core/algorithms/onpolicy_sync/losses/ppo.py)
Defining the PPO loss for actor critic type models.
## PPO
```python
PPO(
    self,
    clip_param: float,
    value_loss_coef: float,
    entropy_coef: float,
    use_clipped_value_loss = True,
    clip_decay: Optional[Callable[[int], float]] = None,
    args,
    kwargs,
)
```
Implementation of the Proximal Policy Optimization loss.

__Attributes__


- `clip_param `: The clipping parameter to use.
- `value_loss_coef `: Weight of the value loss.
- `entropy_coef `: Weight of the entropy (encouraging) loss.
- `use_clipped_value_loss `: Whether or not to also clip the value loss.

## PPOValue
```python
PPOValue(
    self,
    clip_param: float,
    use_clipped_value_loss = True,
    clip_decay: Optional[Callable[[int], float]] = None,
    args,
    kwargs,
)
```
Implementation of the Proximal Policy Optimization loss.

__Attributes__


- `clip_param `: The clipping parameter to use.
- `use_clipped_value_loss `: Whether or not to also clip the value loss.

