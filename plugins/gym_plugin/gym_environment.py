from typing import Optional, Dict, Any, Tuple, List

import gym
import numpy as np


class GymEnvironment:
    def __init__(self, gym_env_name: str):
        self.env = gym.make(gym_env_name)
        self._initial_observation: Optional[np.ndarray] = None
        self.reset()

    @property
    def spec(self) -> gym.envs.registration.EnvSpec:
        return self.env.spec

    @property
    def action_space(self) -> gym.spaces.Space:
        return self.env.action_space

    @property
    def observation_space(self) -> gym.spaces.Space:
        return self.env.observation_space

    @property
    def reward_range(self) -> Tuple[float, float]:
        return self.env.reward_range

    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, Dict]:
        return self.env.step(action)

    def reset(self) -> np.ndarray:
        self._initial_observation = self.env.reset()
        return self._initial_observation

    def render(self, mode="human") -> np.ndarray:
        return self.env.render(mode)

    def close(self):
        self.env.close()

    def seed(self, seed: int = None) -> List[int]:
        return self.env.seed(seed)

    @property
    def unwrapped(self) -> gym.Env:
        return self.env.unwrapped

    @property
    def initial_observation(self) -> np.ndarray:
        assert (
            self._initial_observation is not None
        ), "Attempted to read initial_observation without calling reset()"
        res = self._initial_observation
        self._initial_observation = None
        return res

    def __str__(self) -> str:
        if self.spec is None:
            return "<{} instance>".format(type(self.env).__name__)
        else:
            return "<{}<{}>>".format(type(self.env).__name__, self.spec.id)

    def __exit__(self, *args) -> bool:
        """Support with-statement for the environment. """
        self.env.close()
        # propagate exception
        return False
