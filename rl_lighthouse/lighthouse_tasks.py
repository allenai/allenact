import abc
import string
from typing import List, Dict, Any, Optional, Tuple, Union

import gym
import numpy as np

from rl_base.common import RLStepResult
from rl_base.sensor import Sensor
from rl_base.task import Task, TaskSampler
from rl_lighthouse.lighthouse_environment import LightHouseEnvironment
from utils.experiment_utils import set_seed


class LightHouseTask(Task[LightHouseEnvironment], abc.ABC):
    """Defines an abstract embodied task in the light house gridworld.

    # Attributes

    env : The light house environment.
    sensor_suite: Collection of sensors formed from the `sensors` argument in the initializer.
    task_info : Dictionary of (k, v) pairs defining task goals and other task information.
    max_steps : The maximum number of steps an agent can take an in the task before it is considered failed.
    observation_space: The observation space returned on each step from the sensors.
    """

    def __init__(
        self,
        env: LightHouseEnvironment,
        sensors: List[Sensor],
        task_info: Dict[str, Any],
        max_steps: int,
        **kwargs
    ) -> None:
        """Initializer.

        See class documentation for parameter definitions.
        """
        super().__init__(
            env=env, sensors=sensors, task_info=task_info, max_steps=max_steps, **kwargs
        )

        self._last_action: Optional[int] = None

    @property
    def last_action(self) -> int:
        return self._last_action

    @last_action.setter
    def last_action(self, value: int):
        self._last_action = value

    def step(self, action: int) -> RLStepResult:
        self.last_action = action
        return super(LightHouseTask, self).step(action=action)

    def render(self, mode: str = "array", *args, **kwargs) -> np.ndarray:
        assert mode == "array"

        return self.env.render(mode, *args, **kwargs)


class FindGoalLightHouseTask(LightHouseTask):
    _CACHED_ACTION_NAMES: Dict[int, Tuple[str, ...]] = {}

    def __init__(
        self,
        env: LightHouseEnvironment,
        sensors: List[Sensor],
        task_info: Dict[str, Any],
        max_steps: int,
        **kwargs
    ):
        super().__init__(env, sensors, task_info, max_steps, **kwargs)

        self._found_target = False

    @property
    def action_space(self) -> gym.spaces.Discrete:
        return gym.spaces.Discrete(2 * self.env.world_dim)

    def _step(self, action: int) -> RLStepResult:
        success = self.env.step(action)
        reward = -0.01  # Step penalty
        reward -= (not success) * 0.1  # Failed action penalty

        if np.all(self.env.current_position == self.env.goal_position):
            self._found_target = True
            reward += 1  # Found target reward

        return RLStepResult(
            observation=self.get_observations(),
            reward=reward,
            done=self.is_done(),
            info=None,
        )

    def reached_terminal_state(self) -> bool:
        return self._found_target

    @classmethod
    def class_action_names(cls, world_dim: int = 2, **kwargs) -> Tuple[str, ...]:
        assert 1 <= world_dim <= 26, "Too many dimensions."
        if world_dim not in cls._CACHED_ACTION_NAMES:
            action_names = [
                "{}(+1)".format(string.ascii_lowercase[i] for i in range(world_dim))
            ]
            action_names.extend(
                "{}(-1)".format(string.ascii_lowercase[i] for i in range(world_dim))
            )
            cls._CACHED_ACTION_NAMES[world_dim] = tuple(action_names)

        return cls._CACHED_ACTION_NAMES[world_dim]

    def action_names(self) -> Tuple[str, ...]:
        return self.class_action_names(world_dim=self.env.world_dim)

    def close(self) -> None:
        pass

    def query_expert(self) -> Tuple[Any, bool]:
        raise NotImplementedError()


class FindGoalLightHouseTaskSampler(TaskSampler):
    def __init__(
        self,
        world_dim: int,
        world_radius: int,
        sensors: List[Sensor],
        max_steps: int,
        seed: Optional[int] = None,
        **kwargs
    ):
        self.env = LightHouseEnvironment(world_dim=world_dim, world_radius=world_radius)

        self._last_sampled_task: Optional[FindGoalLightHouseTask] = None
        self.sensors = sensors
        self.max_steps = max_steps

        self.seed: Optional[int] = None
        if seed is not None:
            self.set_seed(seed)

    @property
    def world_dim(self):
        return self.env.world_dim

    @property
    def world_radius(self):
        return self.env.world_radius

    @property
    def __len__(self) -> Union[int, float]:
        return float("inf")

    @property
    def total_unique(self) -> Optional[Union[int, float]]:
        return None

    @property
    def last_sampled_task(self) -> Optional[Task]:
        return self._last_sampled_task

    def next_task(self, force_advance_scene: bool = False) -> Optional[Task]:
        self.env.random_reset()
        return FindGoalLightHouseTask(
            env=self.env, sensors=self.sensors, task_info={}, max_steps=self.max_steps
        )

    def close(self) -> None:
        pass

    @property
    def all_observation_spaces_equal(self) -> bool:
        return True

    def reset(self) -> None:
        raise NotImplementedError

    def set_seed(self, seed: int) -> None:
        set_seed(seed)
        self.seed = seed
