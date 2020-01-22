from typing import NamedTuple, Dict, Any, Type
import copy
import collections.abc


def recursive_update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = recursive_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


class Builder(NamedTuple):
    class_name: Type
    kwargs: Dict[str, Any] = {}
    default: Dict[str, Any] = {}

    def __call__(self, **kwargs):
        allkwargs = copy.deepcopy(self.default)
        recursive_update(allkwargs, self.kwargs)
        recursive_update(allkwargs, kwargs)
        return self.class_name(**allkwargs)
