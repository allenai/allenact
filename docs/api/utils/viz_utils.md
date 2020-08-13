# utils.viz_utils [[source]](https://github.com/allenai/allenact/tree/master/utils/viz_utils.py)

## AbstractViz
```python
AbstractViz(
    self,
    label: Optional[str] = None,
    vector_task_sources: Sequence[Tuple[str, Dict[str, Any]]] = (),
    rollout_sources: Sequence[Union[str, Sequence[str]]] = (),
    actor_critic_source: bool = False,
)
```

### rnn_hidden_memory
Built-in immutable sequence.

If no argument is given, the constructor returns an empty tuple.
If iterable is specified the tuple is initialized from iterable's items.

If the argument is a tuple, the return value is the same object.
### rollout_episode_default_axis
int([x]) -> integer
int(x, base=10) -> integer

Convert a number or string to an integer, or return 0 if no arguments
are given.  If x is a number, return x.__int__().  For floating point
numbers, this truncates towards zero.

If x is not a number or if base is given, then x must be a string,
bytes, or bytearray instance representing an integer literal in the
given base.  The literal can be preceded by '+' or '-' and be surrounded
by whitespace.  The base defaults to 10.  Valid bases are 0 and 2-36.
Base 0 means to interpret the base from the string as an integer literal.
>>> int('0b100', base=0)
4
