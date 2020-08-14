# core.algorithms.onpolicy_sync.losses.imitation [[source]](https://github.com/allenai/allenact/tree/master/core/algorithms/onpolicy_sync/losses/imitation.py)
Defining imitation losses for actor critic type models.
## Imitation
```python
Imitation(self, *args, **kwargs)
```
Expert imitation loss.
### loss
```python
Imitation.loss(
    self,
    step_count: int,
    batch: Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]],
    actor_critic_output: core.base_abstractions.misc.ActorCriticOutput[core.base_abstractions.distributions.CategoricalDistr],
    args,
    kwargs,
)
```
Computes the imitation loss.

__Parameters__


- __batch __: A batch of data corresponding to the information collected when rolling out (possibly many) agents
    over a fixed number of steps. In particular this batch should have the same format as that returned by
    `RolloutStorage.recurrent_generator`.
    Here `batch["observations"]` must contain `"expert_action"` observations
    or `"expert_policy"` observations. See `ExpertActionSensor` (or `ExpertPolicySensor`) for an example of
    a sensor producing such observations.
- __actor_critic_output __: The output of calling an ActorCriticModel on the observations in `batch`.
- __args __: Extra args. Ignored.
- __kwargs __: Extra kwargs. Ignored.

__Returns__


A (0-dimensional) torch.FloatTensor corresponding to the computed loss. `.backward()` will be called on this
tensor in order to compute a gradient update to the ActorCriticModel's parameters.

