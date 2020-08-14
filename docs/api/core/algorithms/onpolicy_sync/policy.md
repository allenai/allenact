# core.algorithms.onpolicy_sync.policy [[source]](https://github.com/allenai/allenact/tree/master/core/algorithms/onpolicy_sync/policy.py)

## ActorCriticModel
```python
ActorCriticModel(
    self,
    action_space: gym.spaces.discrete.Discrete,
    observation_space: gym.spaces.dict.Dict,
)
```
Abstract class defining a deep (recurrent) actor critic agent.

When defining a new agent, you should over subclass this class and implement the abstract methods.

__Attributes__


- `action_space `: The space of actions available to the agent. Currently only discrete
    actions are allowed (so this space will always be of type `gym.spaces.Discrete`).
- `observation_space`: The observation space expected by the agent. This is of type `gym.spaces.dict`.

### forward
```python
ActorCriticModel.forward(
    self,
    args,
    kwargs,
) -> Tuple[core.base_abstractions.misc.ActorCriticOutput[~DistributionType], Optional[torch.Tensor, core.base_abstractions.misc.Memory]]
```
Transforms input observations (& previous hidden state) into action
probabilities and the state value.

__Parameters__


- __args __: extra args.
- __kwargs __: extra kwargs.

__Returns__


A tuple whose first element is an object of class ActorCriticOutput which stores
the agent's probability distribution over possible actions, the agent's value for the
state, and any extra information needed for loss computations. The second element
may be any representation of the agent's hidden states.

### recurrent_hidden_state_size
Non-negative integer corresponding to the dimension of the hidden
state used by the agent or mapping from string memory names to Tuples
of (0) sequences of axes dimensions excluding sampler axis; (1)
position for sampler axis; and (2) data types.

__Returns__


The hidden state dimension (non-negative integer) or dict with memory specification.

## DistributionType
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

