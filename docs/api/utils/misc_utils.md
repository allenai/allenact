# utils.misc_utils [[source]](https://github.com/allenai/allenact/tree/master/utils/misc_utils.py)

## HashableDict
```python
HashableDict(self, *args, **kwargs)
```
A dictionary which is hashable so long as all of its values are
hashable.

A HashableDict object will allow setting / deleting of items until
the first time that `__hash__()` is called on it after which
attempts to set or delete items will throw `RuntimeError`
exceptions.

