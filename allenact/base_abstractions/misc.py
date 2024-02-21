import abc
from typing import (
    Dict,
    Any,
    TypeVar,
    Sequence,
    NamedTuple,
    Optional,
    List,
    Union,
    Generic,
)

import attr
import torch

EnvType = TypeVar("EnvType")
DistributionType = TypeVar("DistributionType")
ModelType = TypeVar("ModelType")
ObservationType = Dict[str, Union[torch.Tensor, Dict[str, Any]]]


class RLStepResult(NamedTuple):
    observation: Optional[Any]
    reward: Optional[Union[float, List[float]]]
    done: Optional[bool]
    info: Optional[Dict[str, Any]]

    def clone(self, new_info: Dict[str, Any]):
        return RLStepResult(
            observation=(
                self.observation
                if "observation" not in new_info
                else new_info["observation"]
            ),
            reward=self.reward if "reward" not in new_info else new_info["reward"],
            done=self.done if "done" not in new_info else new_info["done"],
            info=self.info if "info" not in new_info else new_info["info"],
        )

    def merge(self, other: "RLStepResult"):
        return RLStepResult(
            observation=(
                self.observation if other.observation is None else other.observation
            ),
            reward=self.reward if other.reward is None else other.reward,
            done=self.done if other.done is None else other.done,
            info={
                **(self.info if self.info is not None else {}),
                **(other.info if other is not None else {}),
            },
        )


class ActorCriticOutput(tuple, Generic[DistributionType]):
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
        """Appends a new memory type given its identifier, its memory tensor
        and its sampler dim.

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
        """Returns the memory tensor for a given memory type.

        # Parameters

        key: string identifier of the memory type

        # Returns

        Memory tensor for type `key`
        """
        assert key in self, "Missing key {}".format(key)
        return self[key][0]

    def sampler_dim(self, key: str) -> int:
        """Returns the sampler dimension for the given memory type.

        # Parameters

        key: string identifier of the memory type

        # Returns

        The sampler dim
        """
        assert key in self, "Missing key {}".format(key)
        return self[key][1]

    def sampler_select(self, keep: Sequence[int]) -> "Memory":
        """Equivalent to PyTorch index_select along the `sampler_dim` of each
        memory type.

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
        """Replaces tensor for given key with an updated version.

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
        """Equivalent to slicing with length 1 for the `step` (i.e first)
        dimension in rollouts storage.

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

    def step_squeeze(self, step: int) -> "Memory":
        """Equivalent to simple indexing for the `step` (i.e first) dimension
        in rollouts storage.

        # Parameters

        step: step to keep

        # Returns

        Sliced memory with a single step (and squeezed step dimension)
        """
        res = Memory()
        for key in self:
            tensor = self.tensor(key)
            assert (
                tensor.shape[0] > step
            ), "attempting to access step {} for memory type {} of shape {}".format(
                step, key, tensor.shape
            )
            res.check_append(
                key, self.tensor(key)[step, ...], self.sampler_dim(key) - 1
            )
        return res

    def slice(
        self,
        dim: int,
        start: Optional[int] = None,
        stop: Optional[int] = None,
        step: int = 1,
    ) -> "Memory":
        """Slicing for dimensions that have same extents in all memory types.
        It also accepts negative indices.

        # Parameters

        dim: the dimension to slice
        start: the index of the first item to keep if given (default 0 if None)
        stop: the index of the first item to discard if given (default tensor size along `dim` if None)
        step: the increment between consecutive indices (default 1)

        # Returns

        Sliced memory
        """
        checked = False
        total: Optional[int] = None

        res = Memory()
        for key in self:
            tensor = self.tensor(key)
            assert (
                len(tensor.shape) > dim
            ), f"attempting to access dim {dim} for memory type {key} of shape {tensor.shape}"

            if not checked:
                total = tensor.shape[dim]
                checked = True

            assert (
                total == tensor.shape[dim]
            ), f"attempting to slice along non-uniform dimension {dim}"

            if start is not None or stop is not None or step != 1:
                slice_tuple = (
                    (slice(None),) * dim
                    + (slice(start, stop, step),)
                    + (slice(None),) * (len(tensor.shape) - (1 + dim))
                )
                sliced_tensor = tensor[slice_tuple]
                res.check_append(
                    key=key,
                    tensor=sliced_tensor,
                    sampler_dim=self.sampler_dim(key),
                )
            else:
                res.check_append(
                    key,
                    tensor,
                    self.sampler_dim(key),
                )

        return res

    def to(self, device: torch.device) -> "Memory":
        for key in self:
            tensor = self.tensor(key)
            if tensor.device != device:
                self.set_tensor(key, tensor.to(device))
        return self


class Loss(abc.ABC):
    pass


@attr.s(kw_only=True)
class LossOutput:
    value: torch.Tensor = attr.ib()
    info: Dict[str, Union[float, int]] = attr.ib()
    per_epoch_info: Dict[str, Union[float, int]] = attr.ib()
    batch_memory: Memory = attr.ib()
    stream_memory: Memory = attr.ib()
    bsize: int = attr.ib()


class GenericAbstractLoss(Loss):
    # noinspection PyMethodOverriding
    @abc.abstractmethod
    def loss(  # type: ignore
        self,
        *,  # No positional arguments
        model: ModelType,
        batch: ObservationType,
        batch_memory: Memory,
        stream_memory: Memory,
    ) -> LossOutput:
        """Computes the loss.

        Loss after processing a batch of data with (part of) a model (possibly with memory).

        We support two different types of memory: `batch_memory` and `stream_memory` that can be
        used to compute losses and share computation.

        ## `batch_memory`
        During the update phase of training, the following
        steps happen in order:
        1. A `batch` of data is sampled from an `ExperienceStorage` (which stores data possibly collected during previous
             rollout steps).
        2.  This `batch` is passed to each of the specified `GenericAbstractLoss`'s and is used, along with the `model`,
             to compute each such loss.
        3. The losses are summed together, gradients are computed by backpropagation, and an update step is taken.
        4. The process loops back to (1) with a new batch until.
        Now supposed that the computation used by a `GenericAbstractLoss` (`LossA`) can be shared across multiple of the
        `GenericAbstractLoss`'s (`LossB`, ...). For instance, `LossA` might run the visual encoder of `model` across
        all the images contained in `batch` so that it can compute a classification loss while `LossB` would like to
        run the same visual encoder on the same images to compute a depth-prediction loss. Without having some sort
        of memory, you would need to rerun this visual encoder on all images multiple times, wasting computational
        resources. This is where `batch_memory` comes in: `LossA` is can store the visual representations it computed
        in `batch_memory` and then `LossB` can access them.  Note that the `batch_memory` will be reinitialized after
        each new `batch` is sampled.

        ## `stream_memory`
        As described above, `batch_memory` treats each batch as its own independent collection of data. But what if
        your `ExperienceStorage` samples its batches in a streaming fashion? E.g. your `ExperienceStorage`
        might be a fixed collection of expert trajectories for use with imitation learning. In this case you can't
        simply treat each batch independently: you might want to save information from one batch to use in another.
        The simplest case of this would be if your agent `model` uses an RNN and produces a recurrent hidden state.
        In this case, the hidden state from the end of one batch should be used at the start of computations for the
        next batch. To allow for this, you can use the `stream_memory`. `stream_memory` is not cleared across
        batches but, **importantly**, `stream_memory` is detached from the computation graph after each backpropagation
        step so that the size of the computation graph does not grow unboundedly.

        # Parameters

        model: model to run on data batch (both assumed to be on the same device)
        batch: data to use as input for model (already on the same device as model)
        batch_memory: See above.
        stream_memory: See above.

        # Returns

        A tuple with:

        current_loss: total loss
        current_info: additional information about the current loss
        batch_memory: `batch_memory` memory after processing current data batch, see above.
        stream_memory: `stream_memory` memory after processing current data batch, see above.
        bsize: batch size
        """
        raise NotImplementedError()
