# core.algorithms.offpolicy_sync.losses.abstract_offpolicy_loss [[source]](https://github.com/allenai/embodied-rl/tree/master/core/algorithms/offpolicy_sync/losses/abstract_offpolicy_loss.py)
Defining abstract loss classes for actor critic models.
## AbstractOffPolicyLoss
```python
AbstractOffPolicyLoss(self, *args, **kwargs)
```
Abstract class representing a loss function used to train an
ActorCriticModel.
### loss
```python
AbstractOffPolicyLoss.loss(
    self,
    model: ~ModelType,
    batch: Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]],
    memory: Dict[str, torch.Tensor],
    args,
    kwargs,
) -> Tuple[torch.FloatTensor, Dict[str, float], Dict[str, torch.Tensor]]
```
Computes the loss.

__TODO: Description of how this works__


## ModelType
Type variable.

Usage::

  T = TypeVar('T')  # Can be anything
  A = TypeVar('A', str, bytes)  # Must be str or bytes

Type variables exist primarily for the benefit of static type
checkers.  They serve as the parameters for generic types as well
as for generic function definitions.  See class Generic for more
information on generic types.  Generic functions work as follows:

  def repeat(x: T, n: int) -> List[T]:
      '''Return a list containing n references to x.'''
      return [x]*n

  def longest(x: A, y: A) -> A:
      '''Return the longest of two strings.'''
      return x if len(x) >= len(y) else y

The latter example's signature is essentially the overloading
of (str, str) -> str and (bytes, bytes) -> bytes.  Also note
that if the arguments are instances of some subclass of str,
the return type is still plain str.

At runtime, isinstance(x, T) and issubclass(C, T) will raise TypeError.

Type variables defined with covariant=True or contravariant=True
can be used to declare covariant or contravariant generic types.
See PEP 484 for more details. By default generic types are invariant
in all type variables.

Type variables can be introspected. e.g.:

  T.__name__ == 'T'
  T.__constraints__ == ()
  T.__covariant__ == False
  T.__contravariant__ = False
  A.__constraints__ == (str, bytes)

Note that only type variables defined in global scope can be pickled.

