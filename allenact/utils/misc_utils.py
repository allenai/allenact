import copy
import functools
import hashlib
import inspect
import json
import math
import os
import pdb
import random
import subprocess
import sys
import urllib
import urllib.request
from collections import Counter
from contextlib import contextmanager
from typing import Sequence, List, Optional, Tuple, Hashable

import filelock
import numpy as np
import torch
from scipy.special import comb

from allenact.utils.system import get_logger

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


def multiprocessing_safe_download_file_from_url(url: str, save_path: str):
    with filelock.FileLock(save_path + ".lock"):
        if not os.path.isfile(save_path):
            get_logger().info(f"Downloading file from {url} to {save_path}.")
            urllib.request.urlretrieve(
                url, save_path,
            )
        else:
            get_logger().debug(f"{save_path} exists - skipping download.")


def experimental_api(to_decorate):
    """Decorate a function to note that it is part of the experimental API."""

    have_warned = [False]
    name = f"{inspect.getmodule(to_decorate).__name__}.{to_decorate.__qualname__}"
    if to_decorate.__name__ == "__init__":
        name = name.replace(".__init__", "")

    @functools.wraps(to_decorate)
    def decorated(*args, **kwargs):
        if not have_warned[0]:
            get_logger().warning(
                f"'{name}' is a part of AllenAct's experimental API."
                f" This means: (1) there are likely bugs present and (2)"
                f" we may remove/change this functionality without warning."
                f" USE AT YOUR OWN RISK.",
            )
            have_warned[0] = True
        return to_decorate(*args, **kwargs)

    return decorated


def deprecated(to_decorate):
    """Decorate a function to note that it has been deprecated."""

    have_warned = [False]
    name = f"{inspect.getmodule(to_decorate).__name__}.{to_decorate.__qualname__}"
    if to_decorate.__name__ == "__init__":
        name = name.replace(".__init__", "")

    @functools.wraps(to_decorate)
    def decorated(*args, **kwargs):
        if not have_warned[0]:
            get_logger().warning(
                f"'{name}' has been deprecated and will soon be removed from AllenAct's API."
                f" Please discontinue your use of this function.",
            )
            have_warned[0] = True
        return to_decorate(*args, **kwargs)

    return decorated


class NumpyJSONEncoder(json.JSONEncoder):
    """JSON encoder for numpy objects.

    Based off the stackoverflow answer by Jie Yang here: https://stackoverflow.com/a/57915246.
    The license for this code is [BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
    """

    def default(self, obj):
        if isinstance(obj, np.void):
            return None
        elif isinstance(obj, np.bool_):
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


def unzip(seq: Sequence[Tuple], n: Optional[int]):
    """Undoes a `zip` operation.

    # Parameters

    seq: The sequence of tuples that should be unzipped
    n: The number of items in each tuple. This is an optional value but is necessary if
       `len(seq) == 0` (as there is no other way to infer how many empty lists were zipped together
        in this case) and can otherwise be used to error check.

    # Returns

    A tuple (of length `n` if `n` is given) of lists where the ith list contains all
    the ith elements from the tuples in the input `seq`.
    """
    assert n is not None or len(seq) != 0
    if n is None:
        n = len(seq[0])
    lists = [[] for _ in range(n)]

    for t in seq:
        assert len(t) == n
        for i in range(n):
            lists[i].append(t[i])
    return lists


def uninterleave(seq: Sequence, parts: int) -> List:
    assert 0 < parts <= len(seq)
    n = len(seq)

    quotient = n // parts

    return [
        [seq[i + j * parts] for j in range(quotient + 1) if i + j * parts < len(seq)]
        for i in range(parts)
    ]


@functools.lru_cache(10000)
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


def prepare_locals_for_super(
    local_vars, args_name="args", kwargs_name="kwargs", ignore_kwargs=False
):
    assert (
        args_name not in local_vars
    ), "`prepare_locals_for_super` does not support {}.".format(args_name)
    new_locals = {k: v for k, v in local_vars.items() if k != "self" and "__" not in k}
    if kwargs_name in new_locals:
        if ignore_kwargs:
            new_locals.pop(kwargs_name)
        else:
            kwargs = new_locals.pop(kwargs_name)
            kwargs.update(new_locals)
            new_locals = kwargs
    return new_locals


def partition_limits(num_items: int, num_parts: int):
    return (
        np.round(np.linspace(0, num_items, num_parts + 1, endpoint=True))
        .astype(np.int32)
        .tolist()
    )


def str2bool(v: str):
    v = v.lower().strip()
    if v in ("yes", "true", "t", "y", "1"):
        return True
    elif v in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise ValueError(f"{v} cannot be converted to a bool")


class ForkedPdb(pdb.Pdb):
    """A Pdb subclass that may be used from a forked multiprocessing child."""

    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open("/dev/stdin")
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin
