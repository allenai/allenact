from typing import List, Tuple, Union

import numpy as np

from utils.experiment_utils import EarlyStoppingCriterion, ScalarMeanTracker


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


class StopIfNearOptimal(EarlyStoppingCriterion):
    def __init__(self, optimal: float, deviation: float, min_memory_size: int = 100):
        self.optimal = optimal
        self.deviaion = deviation

        self.current_pos = 0
        self.has_filled = False
        self.memory: np.ndarray = np.zeros(min_memory_size)

    def __call__(
        self,
        stage_steps: int,
        total_steps: int,
        training_metrics: ScalarMeanTracker,
        test_valid_metrics: List[Tuple[str, int, Union[float, np.ndarray]]],
    ) -> bool:
        sums = training_metrics.sums()
        counts = training_metrics.counts()

        k = "ep_length"
        if k in sums:
            count = counts[k]
            ep_length_ave = sums[k] / count

            n = self.memory.shape[0]
            if count >= n:
                if count > n:
                    self.memory = np.full(count, fill_value=ep_length_ave)
                else:
                    self.memory[:] = ep_length_ave
                self.current_pos = 0
                self.has_filled = True
            else:
                self.memory[
                    self.current_pos : (self.current_pos + count)
                ] = ep_length_ave

                if self.current_pos + count > n:
                    self.has_filled = True
                    self.current_pos = self.current_pos + count % n
                    self.memory[: self.current_pos] = ep_length_ave

        if not self.has_filled:
            return False
        return self.memory.mean() < self.optimal + self.deviaion
