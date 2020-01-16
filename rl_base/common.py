import abc
import typing
from typing import Dict, Any, TypeVar

import torch

EnvType = TypeVar("EnvType")
DistributionType = TypeVar("DistributionType")


class RLStepResult(typing.NamedTuple):
    observation: typing.Optional[Any]
    reward: typing.Optional[float]
    done: typing.Optional[bool]
    info: typing.Optional[Dict[str, Any]]

    def clone(self, other: "RLStepResult", new_info: Dict[str]):
        return RLStepResult(
            observation=other.observation
            if "observation" not in new_info
            else new_info["observation"],
            reward=other.reward if "reward" not in new_info else new_info["reward"],
            done=other.done if "done" not in new_info else new_info["done"],
            info=other.info if "info" not in new_info else new_info["info"],
        )

    def merge(self, other: "RLStepResult"):
        return RLStepResult(
            observation=self.observation
            if other.observation is None
            else other.observation,
            reward=self.reward if other.reward is None else other.reward,
            done=self.done if other.done is None else other.done,
            info={
                **(self.info if self.info is not None else {}),
                **(other if other is not None else {}),
            },
        )


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
