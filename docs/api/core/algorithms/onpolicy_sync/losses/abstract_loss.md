# core.algorithms.onpolicy_sync.losses.abstract_loss [[source]](https://github.com/allenai/embodied-rl/tree/master/core/algorithms/onpolicy_sync/losses/abstract_loss.py)
Defining abstract loss classes for actor critic models.
## AbstractActorCriticLoss
```python
AbstractActorCriticLoss(self, *args, **kwargs)
```
Abstract class representing a loss function used to train an
ActorCriticModel.
### loss
```python
AbstractActorCriticLoss.loss(
    self,
    step_count: int,
    batch: Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]],
    actor_critic_output: core.base_abstractions.misc.ActorCriticOutput[core.base_abstractions.distributions.CategoricalDistr],
    args,
    kwargs,
) -> Tuple[torch.FloatTensor, Dict[str, float]]
```
Computes the loss.

__Parameters__


- __batch __: A batch of data corresponding to the information collected when rolling out (possibly many) agents
    over a fixed number of steps. In particular this batch should have the same format as that returned by
    `RolloutStorage.recurrent_generator`.
- __actor_critic_output __: The output of calling an ActorCriticModel on the observations in `batch`.
- __args __: Extra args.
- __kwargs __: Extra kwargs.

__Returns__


A (0-dimensional) torch.FloatTensor corresponding to the computed loss. `.backward()` will be called on this
tensor in order to compute a gradient update to the ActorCriticModel's parameters.

