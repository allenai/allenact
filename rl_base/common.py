import typing
from typing import Dict, Any, TypeVar

EnvType = TypeVar("EnvType")


class RLStepResult(typing.NamedTuple):
    observation: Any
    reward: float
    done: bool
    info: Dict[str, Any]
