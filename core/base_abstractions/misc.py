import abc
import typing
from typing import (
    Dict,
    Any,
    TypeVar,
    Sequence,
    NamedTuple,
    Optional,
    List,
    Union,
    Tuple,
    Callable,
    Hashable,
    cast,
)

import torch

from utils.system import get_logger

EnvType = TypeVar("EnvType")
DistributionType = TypeVar("DistributionType")

KeyType = Union[Hashable, List[Hashable]]
TraversalOperatorType = Callable[["Memory", Hashable, List[Hashable]], Optional[bool]]
StoredTensorAxisType = Tuple[torch.Tensor, int]


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


class Memory(Dict):
    def __init__(self, *args, **kwargs):
        super().__init__()

        self.traversal: Callable[
            ["Memory", TraversalOperatorType, List[Hashable], Optional[Hashable]],
            Optional[bool],
        ] = self.bfs

        if len(args) > 0:
            assert len(args) == 1, (
                "Only one of Sequence[Tuple[Hashable, Tuple[torch.Tensor, int]]]"
                "or Memory accepted as unnamed args"
            )
            if isinstance(args[0], Sequence):
                for key, tensor_dim in args[0]:
                    assert (
                        len(tensor_dim) == 2
                    ), "Only Tuple[torch.Tensor, int] accepted as second item in Tuples"
                    self.check_append(key, *tensor_dim)
            elif isinstance(args[0], Dict):
                self._copy(args[0])
            if len(kwargs) > 0:
                get_logger().warning(
                    "Ignoring kwargs {} when building Memory".format(kwargs)
                )
        elif len(kwargs) > 0:
            for key in kwargs:
                assert isinstance(
                    kwargs[key], tuple
                ), "Only Tuple[torch.Tensor, int]] accepted as keyword arg"
                assert (
                    len(kwargs[key]) == 2
                ), "Only Tuple[torch.Tensor, int]] accepted as keyword arg"
                tensor, dim = kwargs[key]
                self.check_append(key, tensor, dim)

    def _copy(self, source):
        class Copier:
            res = self

            def __call__(self, node, key, path, *args, **kwargs):
                self.res.check_append(path + [key], *node[key])

        func = Copier()
        self.traversal(source, cast(TraversalOperatorType, func))
        return self

    @staticmethod
    def dfs(
        node: "Memory",
        func: TraversalOperatorType,
        path: List[Hashable] = [],
        topkey: Optional[Hashable] = None,
    ) -> Optional[bool]:
        for key in node:
            if topkey is not None:
                assert topkey in node, "Requested missing topkey {}".format(topkey)
                if key != topkey:
                    continue
            if isinstance(node[key], Dict):
                # Do not propagate topkey (traverse all subkeys)
                if Memory.dfs(node[key], func, path + [key]):
                    return True  # terminate traversal
            else:
                if func(node, key, path):
                    return True  # terminate traversal

    @staticmethod
    def bfs(
        node: "Memory",
        func: TraversalOperatorType,
        path: List[Hashable] = [],
        topkey: Optional[Hashable] = None,
    ) -> Optional[bool]:
        to_visit = [(path, node)]
        while len(to_visit) > 0:
            new_to_visit = []
            for path, node in to_visit:
                for key in node:
                    if topkey is not None:
                        assert topkey in node, "Requested missing topkey {}".format(
                            topkey
                        )
                        if key != topkey:
                            continue
                    if isinstance(node[key], Dict):
                        new_to_visit.append((path + [key], node[key]))
                    else:
                        if func(node, key, path):
                            return True  # terminate traversal
            # Do not propagate topkey (traverse all subkeys)
            topkey = None
            to_visit = new_to_visit

    def traverse_to_parent(
        self, key: KeyType, create_children: bool = False
    ) -> Tuple["Memory", Hashable]:
        mem: Memory = self
        if not isinstance(key, Hashable):
            assert isinstance(key, List)
            key = key[::-1]
            # Go to parent key
            while len(key) > 1:
                cur_key = key.pop()
                if cur_key not in mem:
                    if create_children:
                        # create Memory for missing key
                        mem[cur_key] = Memory()
                    else:
                        # inform about first missing key
                        return mem, cast(Hashable, cur_key)
                mem = mem[cur_key]
            # This is the (flat) key within the parent Memory
            key = key[0]
        return mem, cast(Hashable, key)

    def check_append(
        self, key: KeyType, tensor: torch.Tensor, sampler_dim: int,
    ) -> "Memory":
        """Appends a new memory type given its identifier, its memory tensor
        and its sampler dim.

        # Parameters

        key: (path to) string identifier of the memory type
        tensor: memory tensor
        sampler_dim: sampler dimension

        # Returns

        Updated Memory
        """
        assert isinstance(key, Hashable) or isinstance(
            key, List
        ), "key {} must be Hashable or List[Hashable]".format(key)
        assert isinstance(
            tensor, torch.Tensor
        ), "tensor {} must be torch.Tensor".format(tensor)
        assert isinstance(sampler_dim, int), "sampler_dim {} must be int".format(
            sampler_dim
        )

        path = key
        mem, key = self.traverse_to_parent(key, create_children=True)

        assert key not in mem, "Reused key {}".format(path)
        assert (
            0 <= sampler_dim < len(tensor.shape)
        ), "Got sampler_dim {} for tensor with shape {}".format(
            sampler_dim, tensor.shape
        )

        mem[key] = (tensor, sampler_dim)

        return self

    def tensor(self, key: KeyType) -> torch.Tensor:
        """Returns the memory tensor for a given memory identifier.

        # Parameters

        key: (path to) string identifier of the memory type

        # Returns

        Memory tensor for type `key`
        """
        assert isinstance(key, Hashable) or isinstance(
            key, List
        ), "key {} must be Hashable or List[Hashable]".format(key)

        path = key
        mem, key = self.traverse_to_parent(key, create_children=False)

        assert key in mem, "Missing key {}".format(path)
        return mem[key][0]

    def sampler_dim(self, key: KeyType) -> int:
        """Returns the sampler dimension for the given memory identifier.

        # Parameters

        key: (path to) string identifier of the memory type

        # Returns

        The sampler dim
        """
        assert isinstance(key, Hashable) or isinstance(
            key, List
        ), "key {} must be Hashable or List[Hashable]".format(key)

        path = key
        mem, key = self.traverse_to_parent(key, create_children=False)

        assert key in mem, "Missing key {}".format(path)
        return mem[key][1]

    def sampler_select(self, keep: Sequence[int]) -> "Memory":
        """Equivalent to PyTorch index_select along the `sampler_dim` of each
        memory type.

        # Parameters

        keep: a list of sampler indices to keep

        # Returns

        Selected memory
        """

        class SamplerSelector:
            valid = False
            res = Memory()

            def __call__(self, node, key, path, *args, **kwargs):
                sampler_dim = node.sampler_dim(key)
                tensor = node.tensor(key)
                assert len(keep) == 0 or (
                    0 <= min(keep) and max(keep) < tensor.shape[sampler_dim]
                ), "Got min(keep)={} max(keep)={} for memory type {} with shape {}, dim {}".format(
                    min(keep), max(keep), key, tensor.shape, sampler_dim
                )
                # If tensor has size 0 along sampler_dim, never add it to res
                if tensor.shape[sampler_dim] > len(keep):
                    tensor = tensor.index_select(
                        dim=sampler_dim,
                        index=torch.as_tensor(
                            list(keep), dtype=torch.int64, device=tensor.device
                        ),
                    )
                    self.res.check_append(path + [key], tensor, sampler_dim)
                    self.valid = True

        func = SamplerSelector()
        self.traversal(self, cast(TraversalOperatorType, func))
        return func.res if func.valid else self

    def set_tensor(self, key: KeyType, tensor: torch.Tensor) -> "Memory":
        """Replaces tensor for given key with an updated version.

        # Parameters

        key: memory type identifier to update
        tensor: updated tensor

        # Returns

        Updated memory
        """

        # assert key in self, "Missing key {}".format(key)
        assert isinstance(key, Hashable) or isinstance(
            key, List
        ), "key {} must be Hashable or List[Hashable]".format(key)

        path = key
        mem, key = self.traverse_to_parent(key, create_children=False)

        assert key in mem, "Missing key {}".format(path)

        assert (
            tensor.shape == mem[key][0].shape
        ), "setting tensor with shape {} for former {}".format(
            tensor.shape, mem[key][0].shape
        )
        mem[key] = (tensor, mem.sampler_dim(key))

        return self

    def step_select(
        self, step: int, key: Optional[KeyType] = None
    ) -> Union["Memory", StoredTensorAxisType]:
        """Equivalent to slicing with length 1 for the `step` (i.e. first)
        dimension in rollouts storage.

        # Parameters

        step: step to keep

        # Returns

        Sliced memory with a single step
        """

        if key is not None:
            path = key
            mem, key = self.traverse_to_parent(key, create_children=False)

            assert key in mem, "Missing key {}".format(path)
        else:
            mem = self

        class StepSelector:
            res = Memory()

            def __call__(self, node, key, path, *args, **kwargs):
                tensor = node.tensor(key)
                assert (
                    tensor.shape[0] > step
                ), "attempting to access step {} for memory type {} of shape {}".format(
                    step, key, tensor.shape
                )
                if step != -1:
                    self.res.check_append(
                        path + [key],
                        node.tensor(key)[step : step + 1, ...],
                        node.sampler_dim(key),
                    )
                else:
                    self.res.check_append(
                        path + [key],
                        node.tensor(key)[step:, ...],
                        node.sampler_dim(key),
                    )

        func = StepSelector()
        self.traversal(mem, cast(TraversalOperatorType, func), topkey=key)
        return func.res

    def step_squeeze(
        self, step: int, key: Optional[KeyType] = None
    ) -> Union["Memory", StoredTensorAxisType]:
        """Equivalent to simple indexing for the `step` (i.e first) dimension
        in rollouts storage.

        # Parameters

        step: step to keep

        # Returns

        Sliced memory with a single step (and squeezed step dimension)
        """

        if key is not None:
            path = key
            mem, key = self.traverse_to_parent(key, create_children=False)

            assert key in mem, "Missing key {}".format(path)
        else:
            mem = self

        class StepSqueezer:
            res = Memory()

            def __call__(self, node, key, path, *args, **kwargs):
                tensor = node.tensor(key)
                assert (
                    tensor.shape[0] > step
                ), "attempting to access step {} for memory type {} of shape {}".format(
                    step, key, tensor.shape
                )
                self.res.check_append(
                    path + [key],
                    node.tensor(key)[step, ...],
                    node.sampler_dim(key) - 1,
                )

        func = StepSqueezer()
        self.traversal(mem, cast(TraversalOperatorType, func), topkey=key)
        return func.res

    def slice(
        self,
        dim: int,
        start: Optional[int] = None,
        stop: Optional[int] = None,
        step: int = 1,
        key: Optional[KeyType] = None,
    ) -> Union["Memory", StoredTensorAxisType]:
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

        if key is not None:
            path = key
            mem, key = self.traverse_to_parent(key, create_children=False)

            assert key in mem, "Missing key {}".format(path)
        else:
            mem = self

        class Slicer:
            checked = False
            total: Optional[int] = None
            res = Memory()

            def __call__(self, node, key, path, *args, **kwargs):
                tensor = node.tensor(key)
                assert (
                    len(tensor.shape) > dim
                ), "attempting to access dim {} for memory type {} of shape {}".format(
                    dim, key, tensor.shape
                )

                if not self.checked:
                    self.total = tensor.shape[dim]
                    self.checked = True

                assert (
                    self.total == tensor.shape[dim]
                ), "attempting to slice along non-uniform dimension {}".format(dim)

                if start is not None or stop is not None or step != 1:
                    slice_tuple = (
                        (slice(None),) * dim
                        + (slice(start, stop, step),)
                        + (slice(None),) * (len(tensor.shape) - (1 + dim))
                    )
                    sliced_tensor = tensor[slice_tuple]
                    self.res.check_append(
                        key=path + [key],
                        tensor=sliced_tensor,
                        sampler_dim=node.sampler_dim(key),
                    )
                else:
                    self.res.check_append(
                        path + [key], tensor, node.sampler_dim(key),
                    )

        func = Slicer()
        self.traversal(mem, cast(TraversalOperatorType, func), topkey=key)
        return func.res

    # in-place, but returns the original memory views
    def narrow_steps(
        self, num_steps: int, key: Optional[KeyType] = None
    ) -> Optional[Union["Memory", StoredTensorAxisType]]:
        if num_steps == 0:
            get_logger().warning("Attempting to unnarrow with num_steps == 0")
            return None

        if key is not None:
            path = key
            mem, key = self.traverse_to_parent(key, create_children=False)

            assert key in mem, "Missing key {}".format(path)
        else:
            mem = self

        class Narrower:
            origin = key
            res = (
                (Memory(mem[key]) if isinstance(mem[key], Memory) else mem[key])
                if key is not None
                else (Memory(mem) if isinstance(mem, Memory) else mem)
            )

            def __call__(self, node, key, path, *args, **kwargs):
                assert (
                    len(path) == 0 and key == self.origin or path[0] == self.origin
                ) or self.origin is None, (
                    "narrowing unwanted node path {} key {}".format(path, key)
                )
                node[key] = (
                    node.tensor(key).narrow(dim=0, start=0, length=num_steps),
                    node.sampler_dim(key),
                )

        func = Narrower()
        self.traversal(mem, cast(TraversalOperatorType, func), topkey=key)
        return func.res

    def to(self, device: torch.device) -> "Memory":
        class Mover:
            def __call__(self, node, key, *args, **kwargs):
                tensor = node.tensor(key)
                if tensor.device != device:
                    node.set_tensor(key, tensor.to(device))

        func = Mover()
        self.traversal(self, cast(TraversalOperatorType, func))
        return self

    def __setitem__(
        self, key: KeyType, value: Union[Tuple[torch.Tensor, int], "Memory", int]
    ):
        if isinstance(value, Memory):
            # Prevent cyclic graphs by enforcing a deepcopy of
            # the value Memory (still reusing the same tensors)
            super().__setitem__(key, Memory()._copy(value))
        else:
            super().__setitem__(key, value)

    def __len__(self):
        class Counter:
            count = 0

            def __call__(self, *args, **kwargs):
                self.count += 1

        func = Counter()
        self.traversal(self, cast(TraversalOperatorType, func))
        return func.count

    def __contains__(self, key: KeyType):
        mem, key = self.traverse_to_parent(key, create_children=False)
        return super(Memory, mem).__contains__(key)

    def all_tensor_keys(self):
        class Lister:
            paths = []

            def __call__(self, node, key, path, *args, **kwargs):
                self.paths.append(path + [key])

        func = Lister()
        self.traversal(self, cast(TraversalOperatorType, func))
        return func.paths
