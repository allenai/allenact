from typing import Optional, Any

import gym
import numpy as np

from allenact.base_abstractions.sensor import Sensor, prepare_locals_for_super
from allenact.base_abstractions.task import Task, SubTaskType
from allenact_plugins.gym_plugin.gym_environment import GymEnvironment


class GymBox2DSensor(Sensor[gym.Env, Task[gym.Env]]):
    """Wrapper for gym Box2D tasks' observations.
    """

    def __init__(
        self,
        gym_env_name: str = "LunarLanderContinuous-v2",
        uuid: str = "gym_box2d_sensor",
        **kwargs: Any
    ):
        self.gym_env_name = gym_env_name

        observation_space = self._get_observation_space()

        super().__init__(**prepare_locals_for_super(locals()))

    def _get_observation_space(self) -> gym.Space:
        if self.gym_env_name in ["LunarLanderContinuous-v2", "LunarLander-v2"]:
            return gym.spaces.Box(-np.inf, np.inf, shape=(8,), dtype=np.float32)
        elif self.gym_env_name in ["BipedalWalker-v2", "BipedalWalkerHardcore-v2"]:
            high = np.array([np.inf] * 24)
            return gym.spaces.Box(-high, high, dtype=np.float32)
        elif self.gym_env_name == "CarRacing-v0":
            state_w, state_h = 96, 96
            return gym.spaces.Box(
                low=0, high=255, shape=(state_h, state_w, 3), dtype=np.uint8
            )
        raise NotImplementedError()

    def get_observation(
        self,
        env: GymEnvironment,
        task: Optional[SubTaskType],
        *args,
        gym_obs: Optional[np.ndarray] = None,
        **kwargs: Any
    ) -> np.ndarray:
        if gym_obs is not None:
            return gym_obs
        else:
            return env.initial_observation
