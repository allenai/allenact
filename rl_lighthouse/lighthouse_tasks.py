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
        if mode == "array":
            return self.env.render(mode, *args, **kwargs)
        elif mode in ["rgb", "rgb_array", "human"]:
            arr = self.env.render("array", *args, **kwargs)
            colors = np.array(
                [
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
                ],
                dtype=np.uint8,
            )
            return colors[arr]
        else:
            raise NotImplementedError("Render mode '{}' is not supported.".format(mode))


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

        if np.all(self.env.current_position == self.env.goal_position):
            self._found_target = True
            reward += 1  # Found target reward
        elif self.num_steps_taken() == self.max_steps - 1:
            reward = -0.01 / (1 - 0.99)  # TODO: Assumes discounting factor = 0.99

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
        max_tasks: Optional[int] = None,
        seed: Optional[int] = None,
        **kwargs
    ):
        self.env = LightHouseEnvironment(world_dim=world_dim, world_radius=world_radius)

        self._last_sampled_task: Optional[FindGoalLightHouseTask] = None
        self.sensors = sensors
        self.max_steps = max_steps
        self.max_tasks = max_tasks
        self.total_sampled = 0

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
        return float("inf") if self.max_tasks is None else self.max_tasks

    @property
    def total_unique(self) -> Optional[Union[int, float]]:
        return self.max_tasks

    @property
    def last_sampled_task(self) -> Optional[Task]:
        return self._last_sampled_task

    def next_task(self, force_advance_scene: bool = False) -> Optional[Task]:
        if self.max_tasks is not None and self.total_sampled >= self.max_tasks:
            return None

        self.total_sampled += 1
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
        self.total_sampled = 0

    def set_seed(self, seed: int) -> None:
        set_seed(seed)
        self.seed = seed
