# Original work Copyright (c) Facebook, Inc. and its affiliates.
# Modified work Copyright (c) Allen Institute for AI
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Defines the tasks that an agent should complete in a given environment."""
import abc
from abc import abstractmethod
from typing import Dict, Any, Tuple, Generic, Union, List, Optional

import gym
import numpy as np
from gym.spaces.dict import Dict as SpaceDict

from rl_base.common import EnvType, RLStepResult
from rl_base.sensor import Sensor, SensorSuite


class Task(Generic[EnvType]):
    env: EnvType
    sensor_suite: SensorSuite[EnvType]
    task_info: Dict[str, Any]
    max_steps: int
    observation_space: SpaceDict
    action_space: gym.Space

    def __init__(
        self,
        env: EnvType,
        sensors: List[Sensor],
        task_info: Dict[str, Any],
        max_steps: int,
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
    def action_space(self):
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
        @rtype: Dict[str, Any]
        """
        return {}


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
        raise NotImplementedError()

    @abstractmethod
    def next_task(self) -> Task:
        raise NotImplementedError()

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError()

    @property
    @abstractmethod
    def all_observation_spaces_equal(self) -> bool:
        """
        @return: True if all Tasks that can be sampled by this sampler have the
        same observation space. Otherwise False.
        """
        raise NotImplementedError()
