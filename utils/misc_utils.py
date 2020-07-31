import math
import random
import typing
from collections import Counter
from functools import lru_cache
from typing import Sequence, List, Optional

import numpy as np
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


def partition_sequence(input: Sequence, parts: int) -> List:
    assert 0 < parts <= len(input)
    n = len(input)

    quotient = n // parts
    remainder = n % parts
    counts = [quotient + (i < remainder) for i in range(parts)]
    inds = np.cumsum([0] + counts)
    return [input[ind0:ind1] for ind0, ind1 in zip(inds[:-1], inds[1:])]


def uninterleave(input: Sequence, parts: int) -> List:
    assert 0 < parts <= len(input)
    n = len(input)

    quotient = n // parts

    return [
        [
            input[i + j * parts]
            for j in range(quotient + 1)
            if i + j * parts < len(input)
        ]
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

    expected_max = 0
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


def all_equal(s: typing.Sequence):
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
