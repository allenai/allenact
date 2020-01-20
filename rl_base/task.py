# Original work Copyright (c) Facebook, Inc. and its affiliates.
# Modified work Copyright (c) Allen Institute for AI
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Defines the primary data structures by which agents interact with their
environment."""

import abc
from abc import abstractmethod
from typing import Dict, Any, Tuple, Generic, Union, List, Optional

import gym
import numpy as np
from gym.spaces.dict import Dict as SpaceDict

from rl_base.common import EnvType, RLStepResult
from rl_base.sensor import Sensor, SensorSuite


class Task(Generic[EnvType]):
    """An abstract class defining a, goal directed, 'task.' Agents interact
    with their environment through a task by taking a `step` after which they
    receive new observations, rewards, and (potentially) other useful
    information.

    A Task is a helpful generalization of the OpenAI gym's `Env` class
    and allows for multiple tasks (e.g. point and object navigation) to
    be defined on a single environment (e.g. AI2-THOR).
    """

    env: EnvType
    sensor_suite: SensorSuite[EnvType]
    task_info: Dict[str, Any]
    max_steps: int
    observation_space: SpaceDict

    def __init__(
        self,
        env: EnvType,
        sensors: List[Sensor],
        task_info: Dict[str, Any],
        max_steps: int,
        action_space: gym.Space,
        **kwargs
    ) -> None:
        self.env = env
        self.sensor_suite = SensorSuite(sensors)
        self.task_info = task_info
        self.max_steps = max_steps
        self.observation_space = SpaceDict(
            {**self.sensor_suite.observation_spaces.spaces,}
        )
        self._num_steps_taken = 0

    def get_observations(self) -> Any:
        return self.sensor_suite.get_observations(env=self.env, task=self)

    @property
    @abstractmethod
    def action_space(self) -> gym.Space:
        """
        @return: the action space for the task.
        """
        raise NotImplementedError()

    @abstractmethod
    def render(self, mode: str = "rgb", *args, **kwargs) -> np.ndarray:
        raise NotImplementedError()

    def _increment_num_steps_taken(self) -> None:
        self._num_steps_taken += 1

    def step(self, action: int) -> RLStepResult:
        assert not self.is_done()
        self._increment_num_steps_taken()
        return self._step(action=action)

    @abstractmethod
    def _step(self, action: int) -> RLStepResult:
        raise NotImplementedError()

    def reached_max_steps(self) -> bool:
        return self.num_steps_taken() >= self.max_steps

    @abstractmethod
    def reached_terminal_state(self) -> bool:
        raise NotImplementedError()

    def is_done(self) -> bool:
        return self.reached_terminal_state() or self.reached_max_steps()

    def num_steps_taken(self) -> int:
        return self._num_steps_taken

    def info(self):
        return {"ep_length": self.num_steps_taken()}

    @classmethod
    @abstractmethod
    def action_names(cls) -> Tuple[str, ...]:
        """
        @return: tuple of (ordered) action names so that taking action
            running `task.step(i)` corresponds to taking action task.action_names()[i].
        """
        raise NotImplementedError()

    @property
    def total_actions(self) -> int:
        return len(self.action_names())

    def index_to_action(self, index: int) -> str:
        assert 0 <= index < self.total_actions
        return self.action_names()[index]

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError()

    def metrics(self) -> Dict[str, Any]:
        """Computes metrics related to the task after the task's completion.

        @return: a dictionary where every key is a string (the metric's
            name) and the value is the value of the metric.
        """
        return {}

    def query_expert(self) -> Tuple[Any, bool]:
        """
        @return: a tuple (x, y) where x is the expert action (or policy) and y is False \
            if the expert could not determine the optimal action (otherwise True). Here y \
            is used for masking. Even when y is False, x should still lie in the space of \
            possible values (e.g. if x is the expert policy then x should be the correct length, \
            sum to 1, and have non-negative entries).
        """
        raise NotImplementedError()


class TaskSampler(abc.ABC):
    """Abstract class defining a how new tasks are sampled."""

    @property
    @abstractmethod
    def __len__(self) -> Union[int, float]:
        """
        @return: Number of total tasks remaining that can be sampled. Can be
            float('inf').
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def total_unique(self) -> Union[int, float, None]:
        """
        @return: Total number of *unique* tasks that can be sampled. Can be
            float('inf') or, if the total unique is not known, None.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def last_sampled_task(self) -> Optional[Task]:
        """
        @return: the most recently sampled Task.
        """
        raise NotImplementedError()

    @abstractmethod
    def next_task(self) -> Task:
        """
        @return: the next Task in the sampler's stream.
        """
        raise NotImplementedError()

    @abstractmethod
    def close(self) -> None:
        """Closes any open environments or streams.

        Should be run when done sampling.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def all_observation_spaces_equal(self) -> bool:
        """
        @return: True if all Tasks that can be sampled by this sampler have the
            same observation space. Otherwise False.
        """
        raise NotImplementedError()
