import abc
import typing
from typing import Dict, Any, TypeVar, Sequence, Tuple, NamedTuple, Optional

import torch

from utils.system import LOGGER

EnvType = TypeVar("EnvType")
DistributionType = TypeVar("DistributionType")


class RLStepResult(NamedTuple):
    observation: Optional[Any]
    reward: Optional[float]
    done: Optional[bool]
    info: Optional[Dict[str, Any]]

    def clone(self, new_info: Dict[str, Any]):
        return RLStepResult(
            observation=self.observation
            if "observation" not in new_info
            else new_info["observation"],
            reward=self.reward if "reward" not in new_info else new_info["reward"],
            done=self.done if "done" not in new_info else new_info["done"],
            info=self.info if "info" not in new_info else new_info["info"],
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
                **(other.info if other is not None else {}),
            },
        )


class ActorCriticOutput(tuple, typing.Generic[DistributionType]):
    distributions: DistributionType
    values: torch.FloatTensor
    extras: Dict[str, Any]

    # noinspection PyTypeChecker
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
    def __init__(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def loss(self, *args, **kwargs):
        raise NotImplementedError()


class Memory(Dict):
    def __init__(self, *args, **kwargs):
        super().__init__()
        if len(args) > 0:
            assert len(args) == 1, "Only 1 Sequence[Tuple[str, Tuple[torch.Tensor, int]]]" \
                                   "or Dict[str, Tuple[torch.Tensor, int]] accepted as unnamed args"
            if isinstance(args[0], Sequence):
                for key, tensor_dim in args[0]:
                    assert len(tensor_dim) == 2, "Only Tuple[torch.Tensor, int]] accepted as second item in Tuples"
                    tensor, dim = tensor_dim
                    self.check_append(key, tensor, dim)
            elif isinstance(args[0], Dict):
                for key in args[0]:
                    assert len(args[0][key]) == 2, "Only Tuple[torch.Tensor, int]] accepted as values in Dict"
                    tensor, dim = args[0][key]
                    self.check_append(key, tensor, dim)
        elif len(kwargs) > 0:
            for key in kwargs:
                assert len(kwargs[key]) == 2, "Only Tuple[torch.Tensor, int]] accepted as keyword arg"
                tensor, dim = kwargs[key]
                self.check_append(key, tensor, dim)

    def check_append(self, key: str, tensor: torch.Tensor, sampler_dim: int):
        assert isinstance(key, str), "key {} must be str".format(key)
        assert isinstance(tensor, torch.Tensor), "tensor {} must be torch.Tensor".format(tensor)
        assert isinstance(sampler_dim, int), "sampler_dim {} must be int".format(sampler_dim)

        assert key not in self, "Reused key {}".format(key)
        assert 0 <= sampler_dim < len(tensor.shape),\
            "Got sampler_dim {} for tensor with shape {}".format(sampler_dim, tensor.shape)

        self[key] = (tensor, sampler_dim)

    def tensor(self, key: str):
        assert key in self, "Missing key {}".format(key)
        return self[key][0]

    def sampler_dim(self, key: str):
        assert key in self, "Missing key {}".format(key)
        return self[key][1]

    def index_select(self, keep: Sequence[int]):
        res = Memory()
        for name in self:
            sampler_dim = self.sampler_dim(name)
            tensor = self.tensor(name)
            if tensor.shape[sampler_dim] > len(keep):
                tensor = tensor.index_select(
                    dim=sampler_dim,
                    index=torch.as_tensor(list(keep), dtype=torch.int64, device=tensor.device)
                )
                res.check_append(name, tensor, sampler_dim)
        return res
