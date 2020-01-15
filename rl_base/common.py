import abc
import typing
from typing import Dict, Any, TypeVar

import torch

EnvType = TypeVar("EnvType")
DistributionType = TypeVar("DistributionType")


class RLStepResult(typing.NamedTuple):
    observation: Any
    reward: float
    done: bool
    info: Dict[str, Any]

class ActorCriticOutput(tuple, typing.Generic[DistributionType]):
    distributions: DistributionType
    values: torch.FloatTensor
    extras: Dict[str, Any]

    def __new__(
        cls,
        distributions: DistributionType,
        values: torch.FloatTensor,
        extras: Dict[str, Any],
    ):
        self = tuple.__new__(cls, (distributions, values, extras))
        self.distributions = distributions
        self.values = values
        self.extras = extras
        return self

    def __repr__(self) -> str:
        return (
            f"Group(distributions={self.distributions},"
            f" values={self.values},"
            f" extras={self.extras})"
        )


class Loss(abc.ABC):
    @abc.abstractmethod
    def loss(self, *args, **kwargs):
        raise NotImplementedError()
