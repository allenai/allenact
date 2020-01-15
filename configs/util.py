from typing import NamedTuple, Dict, Any, Optional


class ClassKWArgsTuple(NamedTuple):
    cname: Any
    kwargs: Optional[Dict] = {}
