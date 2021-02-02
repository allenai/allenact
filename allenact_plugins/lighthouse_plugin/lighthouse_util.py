from typing import List, Tuple, Union

import numpy as np

from allenact.utils.experiment_utils import EarlyStoppingCriterion, ScalarMeanTracker


class StopIfNearOptimal(EarlyStoppingCriterion):
    def __init__(self, optimal: float, deviation: float, min_memory_size: int = 100):
        self.optimal = optimal
        self.deviation = deviation

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
                    # Increase memory size to fit all of the new values
                    self.memory = np.full(count, fill_value=ep_length_ave)
                else:
                    # We have exactly as many values as the memory size,
                    # simply set the whole memory to be equal to the new
                    # average ep length.
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

        return self.memory.mean() < self.optimal + self.deviation
