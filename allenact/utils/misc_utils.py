import hashlib
import json

import copy
import hashlib
import math
import random
import subprocess
from collections import Counter
from contextlib import contextmanager
from functools import lru_cache
from typing import Sequence, List, Optional, Tuple, Hashable

import numpy as np
import torch
from scipy.special import comb

TABLEAU10_RGB = (
    (31, 119, 180),
    (255, 127, 14),
    (44, 160, 44),
    (214, 39, 40),
    (148, 103, 189),
    (140, 86, 75),
    (227, 119, 194),
    (127, 127, 127),
    (188, 189, 34),
    (23, 190, 207),
)


class NumpyJSONEncoder(json.JSONEncoder):
    """JSON encoder for numpy objects.

    Based off the stackoverflow answer by Jie Yang here: https://stackoverflow.com/a/57915246.
    The license for this code is [BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
    """

    def default(self, obj):
        if isinstance(obj, np.void):
            return None
        elif isinstance(obj, np.bool):
            return bool(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyJSONEncoder, self).default(obj)


@contextmanager
def tensor_print_options(**print_opts):
    torch_print_opts = copy.deepcopy(torch._tensor_str.PRINT_OPTS)
    np_print_opts = np.get_printoptions()
    try:
        torch.set_printoptions(**print_opts)
        np.set_printoptions(**print_opts)
        yield None
    finally:
        torch.set_printoptions(**{k: getattr(torch_print_opts, k) for k in print_opts})
        np.set_printoptions(**np_print_opts)


def md5_hash_str_as_int(to_hash: str):
    return int(hashlib.md5(to_hash.encode()).hexdigest(), 16,)


def get_git_diff_of_project() -> Tuple[str, str]:
    short_sha = (
        subprocess.check_output(["git", "describe", "--always"]).decode("utf-8").strip()
    )
    diff = subprocess.check_output(["git", "diff", short_sha]).decode("utf-8")
    return short_sha, diff


class HashableDict(dict):
    """A dictionary which is hashable so long as all of its values are
    hashable.

    A HashableDict object will allow setting / deleting of items until
    the first time that `__hash__()` is called on it after which
    attempts to set or delete items will throw `RuntimeError`
    exceptions.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._hash_has_been_called = False

    def __key(self):
        return tuple((k, self[k]) for k in sorted(self))

    def __hash__(self):
        self._hash_has_been_called = True
        return hash(self.__key())

    def __eq__(self, other):
        return self.__key() == other.__key()

    def __setitem__(self, *args, **kwargs):
        if not self._hash_has_been_called:
            return super(HashableDict, self).__setitem__(*args, **kwargs)
        raise RuntimeError("Cannot set item in HashableDict after having called hash.")

    def __delitem__(self, *args, **kwargs):
        if not self._hash_has_been_called:
            return super(HashableDict, self).__delitem__(*args, **kwargs)
        raise RuntimeError(
            "Cannot delete item in HashableDict after having called hash."
        )


def partition_sequence(seq: Sequence, parts: int) -> List:
    assert 0 < parts, f"parts [{parts}] must be greater > 0"
    assert parts <= len(seq), f"parts [{parts}] > len(seq) [{len(seq)}]"
    n = len(seq)

    quotient = n // parts
    remainder = n % parts
    counts = [quotient + (i < remainder) for i in range(parts)]
    inds = np.cumsum([0] + counts)
    return [seq[ind0:ind1] for ind0, ind1 in zip(inds[:-1], inds[1:])]


def uninterleave(seq: Sequence, parts: int) -> List:
    assert 0 < parts <= len(seq)
    n = len(seq)

    quotient = n // parts

    return [
        [seq[i + j * parts] for j in range(quotient + 1) if i + j * parts < len(seq)]
        for i in range(parts)
    ]


@lru_cache(10000)
def cached_comb(n: int, m: int):
    return comb(n, m)


def expected_max_of_subset_statistic(vals: List[float], m: int):
    n = len(vals)
    assert m <= n

    vals_and_counts = list(Counter([round(val, 8) for val in vals]).items())
    vals_and_counts.sort()

    count_so_far = 0
    logdenom = math.log(comb(n, m))

    expected_max = 0.0
    for val, num_occurances_of_val in vals_and_counts:
        count_so_far += num_occurances_of_val
        if count_so_far < m:
            continue

        count_where_max = 0
        for i in range(1, min(num_occurances_of_val, m) + 1):
            count_where_max += cached_comb(num_occurances_of_val, i) * cached_comb(
                count_so_far - num_occurances_of_val, m - i
            )

        expected_max += val * math.exp(math.log(count_where_max) - logdenom)

    return expected_max


def bootstrap_max_of_subset_statistic(
    vals: List[float], m: int, reps=1000, seed: Optional[int] = None
):
    rstate = None
    if seed is not None:
        rstate = random.getstate()
        random.seed(seed)
    results = []
    for _ in range(reps):
        results.append(
            expected_max_of_subset_statistic(random.choices(vals, k=len(vals)), m)
        )

    if seed is not None:
        random.setstate(rstate)
    return results


def rand_float(low: float, high: float, shape):
    assert low <= high
    try:
        return np.random.rand(*shape) * (high - low) + low
    except TypeError as _:
        return np.random.rand(shape) * (high - low) + low


def all_unique(seq: Sequence[Hashable]):
    seen = set()
    for s in seq:
        if s in seen:
            return False
        seen.add(s)
    return True


def all_equal(s: Sequence):
    if len(s) <= 1:
        return True
    return all(s[0] == ss for ss in s[1:])


def prepare_locals_for_super(local_vars, args_name="args", kwargs_name="kwargs"):
    assert (
        args_name not in local_vars
    ), "`prepare_locals_for_super` does not support {}.".format(args_name)
    new_locals = {k: v for k, v in local_vars.items() if k != "self" and "__" not in k}
    if kwargs_name in new_locals:
        kwargs = new_locals[kwargs_name]
        del new_locals[kwargs_name]
        kwargs.update(new_locals)
        new_locals = kwargs
    return new_locals


def partition_limits(num_items: int, num_parts: int):
    return (
        np.round(np.linspace(0, num_items, num_parts + 1, endpoint=True))
        .astype(np.int32)
        .tolist()
    )
