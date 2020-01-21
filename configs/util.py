from typing import NamedTuple, Dict, Any, Type
import copy
import collections.abc


class Builder(NamedTuple):
    class_name: Type
    kwargs: Dict[str, Any] = {}
    default: Dict[str, Any] = {}

    def update(self, d, u):
        for k, v in u.items():
            if isinstance(v, collections.abc.Mapping):
                d[k] = self.update(d.get(k, {}), v)
            else:
                d[k] = v
        return d

    def __call__(self, **kwargs):
        allkwargs = copy.deepcopy(self.default)
        self.update(allkwargs, self.kwargs)
        self.update(allkwargs, kwargs)
        return self.class_name(**allkwargs)
