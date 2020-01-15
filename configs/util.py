import copy
from typing import NamedTuple, Dict, Any


class Builder(NamedTuple):
    class_name: Any
    kwargs: Dict[str, Any] = {}
    default: Dict[str, Any] = {}

    def __call__(self, **kwargs):
        allkwargs = copy.deepcopy(self.default)
        allkwargs.update(self.kwargs)
        allkwargs.update(kwargs)
        return self.class_name(**allkwargs)
