# Original work Copyright (c) Facebook, Inc. and its affiliates.
# Modified work Copyright (c) Allen Institute for AI
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Defines the tasks that an agent should complete in a given environment."""
import abc
from abc import abstractmethod
from typing import Dict, Any, Tuple, Generic, Union, List

from rl_base.common import EnvType, RLStepResult
from rl_base.sensor import Sensor, SensorSuite


class Task(Generic[EnvType]):
    env: EnvType
    sensors: SensorSuite[EnvType]
    task_info: Dict[str, Any]
    max_steps: int

    def __init__(
        self,
        env: EnvType,
        sensors: List[Sensor],
        task_info: Dict[str, Any],
        max_steps: int,
        **kwargs
    ) -> None:
        self.env = env
        self.sensors = SensorSuite(sensors)
        self.task_info = task_info
        self.max_steps = max_steps

        self._num_steps_taken = 0

    @abstractmethod
    def get_observations(self) -> Any:
        return self.sensors.get_observations(self.env)

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
    def action_names(self) -> Tuple[str, ...]:
        raise NotImplementedError()

    @property
    def total_actions(self) -> int:
        return len(self.action_names())

    def index_to_action(self, index: int) -> str:
        assert 0 <= index < self.total_actions
        return self.action_names()[index]


class TaskSampler(abc.ABC):
    # Abstract class defining a how new tasks are sampled

    @property
    def __len__(self) -> Union[int, float]:
        r"""
        Returns:
            Number of total tasks remaining that can be sampled.
            Can be float('inf').

        """
        raise NotImplementedError()

    @property
    def total_unique(self) -> Union[int, float, None]:
        r"""
        Returns:
            Total number of *unique* tasks that can be sampled.
            Can be float('inf') or, if the total unique is not known,
            None.
        """
        raise NotImplementedError()

    def next_task(self) -> Task:
        raise NotImplementedError()
