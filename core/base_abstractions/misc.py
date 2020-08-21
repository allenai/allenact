import abc
import typing
from typing import Dict, Any, TypeVar, Sequence, NamedTuple, Optional, List, Union

import torch

EnvType = TypeVar("EnvType")
DistributionType = TypeVar("DistributionType")


class RLStepResult(NamedTuple):
    observation: Optional[Any]
    reward: Optional[Union[float, List[float]]]
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


# TODO document that the step dim is always 0 (as in policy's memory specification)
class Memory(Dict):
    def __init__(self, *args, **kwargs):
        super().__init__()
        if len(args) > 0:
            assert len(args) == 1, (
                "Only one of Sequence[Tuple[str, Tuple[torch.Tensor, int]]]"
                "or Dict[str, Tuple[torch.Tensor, int]] accepted as unnamed args"
            )
            if isinstance(args[0], Sequence):
                for key, tensor_dim in args[0]:
                    assert (
                        len(tensor_dim) == 2
                    ), "Only Tuple[torch.Tensor, int]] accepted as second item in Tuples"
                    tensor, dim = tensor_dim
                    self.check_append(key, tensor, dim)
            elif isinstance(args[0], Dict):
                for key in args[0]:
                    assert (
                        len(args[0][key]) == 2
                    ), "Only Tuple[torch.Tensor, int]] accepted as values in Dict"
                    tensor, dim = args[0][key]
                    self.check_append(key, tensor, dim)
        elif len(kwargs) > 0:
            for key in kwargs:
                assert (
                    len(kwargs[key]) == 2
                ), "Only Tuple[torch.Tensor, int]] accepted as keyword arg"
                tensor, dim = kwargs[key]
                self.check_append(key, tensor, dim)

    def check_append(
        self, key: str, tensor: torch.Tensor, sampler_dim: int
    ) -> "Memory":
        """
        Appends a new memory type given its identifier, its memory tensor and its sampler dim.

        # Parameters

        key: string identifier of the memory type
        tensor: memory tensor
        sampler_dim: sampler dimension

        # Returns

        Updated Memory
        """
        assert isinstance(key, str), "key {} must be str".format(key)
        assert isinstance(
            tensor, torch.Tensor
        ), "tensor {} must be torch.Tensor".format(tensor)
        assert isinstance(sampler_dim, int), "sampler_dim {} must be int".format(
            sampler_dim
        )

        assert key not in self, "Reused key {}".format(key)
        assert (
            0 <= sampler_dim < len(tensor.shape)
        ), "Got sampler_dim {} for tensor with shape {}".format(
            sampler_dim, tensor.shape
        )

        self[key] = (tensor, sampler_dim)

        return self

    def tensor(self, key: str) -> torch.Tensor:
        """
        Returns the memory tensor for a given memory type.

        # Parameters

        key: string identifier of the memory type

        # Returns

        Memory tensor for type `key`
        """
        assert key in self, "Missing key {}".format(key)
        return self[key][0]

    def sampler_dim(self, key: str) -> int:
        """
        Returns the sampler dimension for the given memory type.

        # Parameters

        key: string identifier of the memory type

        # Returns

        The sampler dim
        """
        assert key in self, "Missing key {}".format(key)
        return self[key][1]

    def sampler_select(self, keep: Sequence[int]) -> "Memory":
        """
        Equivalent to PyTorch index_select along the `sampler_dim` of each memory type.

        # Parameters

        keep: a list of sampler indices to keep

        # Returns

        Selected memory
        """
        res = Memory()
        valid = False
        for name in self:
            sampler_dim = self.sampler_dim(name)
            tensor = self.tensor(name)
            assert len(keep) == 0 or (
                0 <= min(keep) and max(keep) < tensor.shape[sampler_dim]
            ), "Got min(keep)={} max(keep)={} for memory type {} with shape {}, dim {}".format(
                min(keep), max(keep), name, tensor.shape, sampler_dim
            )
            if tensor.shape[sampler_dim] > len(keep):
                tensor = tensor.index_select(
                    dim=sampler_dim,
                    index=torch.as_tensor(
                        list(keep), dtype=torch.int64, device=tensor.device
                    ),
                )
                res.check_append(name, tensor, sampler_dim)
                valid = True
        if valid:
            return res
        return self

    def set_tensor(self, key: str, tensor: torch.Tensor) -> "Memory":
        """
        Replaces tensor for given key with an updated version

        # Parameters

        key: memory type identifier to update
        tensor: updated tensor

        # Returns

        Updated memory
        """
        assert key in self, "Missing key {}".format(key)
        assert (
            tensor.shape == self[key][0].shape
        ), "setting tensor with shape {} for former {}".format(
            tensor.shape, self[key][0].shape
        )
        self[key] = (tensor, self[key][1])

        return self

    def step_select(self, step: int) -> "Memory":
        """
        Equivalent to slicing with length 1 for the `step` (i.e first) dimension.

        # Parameters

        step: step to keep

        # Returns

        Sliced memory with a single step
        """
        res = Memory()
        for key in self:
            tensor = self.tensor(key)
            assert (
                tensor.shape[0] > step
            ), "attempting to access step {} for memory type {} of shape {}".format(
                step, key, tensor.shape
            )
            if step != -1:
                res.check_append(
                    key, self.tensor(key)[step : step + 1, ...], self.sampler_dim(key)
                )
            else:
                res.check_append(
                    key, self.tensor(key)[step:, ...], self.sampler_dim(key)
                )
        return res

    def slice(
        self, dim: int, first: Optional[int] = None, last: Optional[int] = None
    ) -> "Memory":
        """
        Slicing for dimensions that have same extents in all memory types. It also accepts negative indices.

        # Parameters

        dim: the dimension to slice
        first: the index of the first item to keep if given (default 0 if None)
        last: the index of the first item to discard if given (default tensor shape along `dim` if None)

        # Returns

        Sliced memory
        """
        if first is None:
            first = 0
        checked = False
        total: Optional[int] = None
        index: Optional[torch.Tensor] = None

        res = Memory()
        for key in self:
            tensor = self.tensor(key)
            assert (
                len(tensor.shape) > dim
            ), "attempting to access dim {} for memory type {} of shape {}".format(
                dim, key, tensor.shape
            )

            if not checked:
                total = tensor.shape[dim]
                if first < 0:
                    first += total
                if last is None:
                    last = total
                elif last < 0:
                    last += total

                assert (
                    0 <= first <= last <= total
                ), "attempting to slice with first {} last {} for {} elems".format(
                    first, last, total
                )

                # assume all tensors are in the same device
                if last - first < total:
                    index = torch.as_tensor(
                        list(range(first, last)),
                        dtype=torch.int64,
                        device=tensor.device,
                    )

                checked = True

            assert (
                total == tensor.shape[dim]
            ), "attempting to slice along non-uniform dimension {}".format(dim)

            if last - first < total:
                res.check_append(
                    key,
                    tensor.index_select(dim=dim, index=index),
                    self.sampler_dim(key),
                )
            else:
                res.check_append(
                    key, tensor, self.sampler_dim(key),
                )

        return res
