from typing import NamedTuple, Dict, Any, Type, Union
import copy
import collections.abc


def recursive_update(
    original: Union[Dict, collections.abc.MutableMapping],
    update: Union[Dict, collections.abc.MutableMapping],
):
    """Recursively updates original dictionary with entries form update dict.

    Args:
        original:  Original dictionary to be updated.
        update:    Dictionary with additional or replacement entries

    Returns:
        Updated original dictionary.
    """
    for k, v in update.items():
        if isinstance(v, collections.abc.MutableMapping):
            original[k] = recursive_update(original.get(k, {}), v)
        else:
            original[k] = v
    return original


class Builder(NamedTuple):
    class_name: Type
    kwargs: Dict[str, Any] = {}
    default: Dict[str, Any] = {}

    def __call__(self, **kwargs):
        allkwargs = copy.deepcopy(self.default)
        recursive_update(allkwargs, self.kwargs)
        recursive_update(allkwargs, kwargs)
        return self.class_name(**allkwargs)
