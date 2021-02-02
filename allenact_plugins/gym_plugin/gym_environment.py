from typing import Optional

import gym
import numpy as np


class GymEnvironment(gym.Wrapper):
    """gym.Wrapper with minimal bookkeeping (initial observation)."""

    def __init__(self, gym_env_name: str):
        super().__init__(gym.make(gym_env_name))
        self._initial_observation: Optional[np.ndarray] = None
        self.reset()  # generate initial observation

    def reset(self) -> np.ndarray:
        self._initial_observation = self.env.reset()
        return self._initial_observation

    @property
    def initial_observation(self) -> np.ndarray:
        assert (
            self._initial_observation is not None
        ), "Attempted to read initial_observation without calling reset()"
        res = self._initial_observation
        self._initial_observation = None
        return res
