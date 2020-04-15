# Original work Copyright (c) Facebook, Inc. and its affiliates.
# Modified work Copyright (c) Allen Institute for AI
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Defines the primary data structures by which agents interact with their
environment."""

import abc
from abc import abstractmethod
from typing import Dict, Any, Tuple, Generic, Union, List, Optional, TypeVar

import gym
import numpy as np
from gym.spaces.dict import Dict as SpaceDict

from rl_base.common import RLStepResult
from rl_base.sensor import Sensor, SensorSuite

EnvType = TypeVar("EnvType")


class Task(Generic[EnvType]):
    """An abstract class defining a, goal directed, 'task.' Agents interact
    with their environment through a task by taking a `step` after which they
    receive new observations, rewards, and (potentially) other useful
    information.

    A Task is a helpful generalization of the OpenAI gym's `Env` class
    and allows for multiple tasks (e.g. point and object navigation) to
    be defined on a single environment (e.g. AI2-THOR).

    # Attributes

    env : The environment.
    sensor_suite: Collection of sensors formed from the `sensors` argument in the initializer.
    task_info : Dictionary of (k, v) pairs defining task goals and other task information.
    max_steps : The maximum number of steps an agent can take an in the task before it is considered failed.
    observation_space: The observation space returned on each step from the sensors.
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
        self._total_reward = 0.0

    def get_observations(self) -> Any:
        return self.sensor_suite.get_observations(env=self.env, task=self)

    @property
    @abstractmethod
    def action_space(self) -> gym.Space:
        """Task's action space.

        # Returns

        The action space for the task.
        """
        raise NotImplementedError()

    @abstractmethod
    def render(self, mode: str = "rgb", *args, **kwargs) -> np.ndarray:
        """Render the current task state.

        Rendered task state can come in any supported modes.

        # Parameters

        mode : The mode in which to render. For example, you might have a 'rgb'
            mode that renders the agent's egocentric viewpoint or a 'dev' mode
            returning additional information.
        args : Extra args.
        kwargs : Extra kwargs.

        # Returns

        An numpy array corresponding to the requested render.
        """
        raise NotImplementedError()

    def _increment_num_steps_taken(self) -> None:
        """Helper function that increases the number of steps counter by
        one."""
        self._num_steps_taken += 1

    def step(self, action: int) -> RLStepResult:
        """Take an action in the environment.

        Takes the action in the environment corresponding to
        `self.action_names()[action]` and returns
        observations (& rewards and any additional information)
        corresponding to the agent's new state. Note that this function
        should not be overwritten without care (instead
        implement the `_step` function).

        # Parameters

        action : The action to take.

        # Returns

        A `RLStepResult` object encoding the new observations, reward, and
        (possibly) additional information.
        """
        assert not self.is_done()
        sr = self._step(action=action)
        self._total_reward += float(sr.reward)
        self._increment_num_steps_taken()
        return sr

    @abstractmethod
    def _step(self, action: int) -> RLStepResult:
        """Helper function called by `step` to take a step in the environment.

        Takes the action in the environment corresponding to
        `self.action_names()[action]` and returns
        observations (& rewards and any additional information)
        corresponding to the agent's new state. This function is called
        by the (public) `step` function and is what should be implemented
        when defining your new task. Having separate `_step` be separate from `step`
        is useful as this allows the `step` method to perform bookkeeping (e.g.
        keeping track of the number of steps), without having `_step` as a separate
        method, everyone implementing `step` would need to copy this bookkeeping code.

        # Parameters

        action : The action to take.

        # Returns

        A `RLStepResult` object encoding the new observations, reward, and
        (possibly) additional information.
        """
        raise NotImplementedError()

    def reached_max_steps(self) -> bool:
        """Has the agent reached the maximum number of steps."""
        return self.num_steps_taken() >= self.max_steps

    @abstractmethod
    def reached_terminal_state(self) -> bool:
        """Has the agent reached a terminal state (excluding reaching the
        maximum number of steps)."""
        raise NotImplementedError()

    def is_done(self) -> bool:
        """Did the agent reach a terminal state or performed the maximum number
        of steps."""
        return self.reached_terminal_state() or self.reached_max_steps()

    def num_steps_taken(self) -> int:
        """Number of steps taken by the agent in the task so far."""
        return self._num_steps_taken

    @classmethod
    @abstractmethod
    def action_names(cls) -> Tuple[str, ...]:
        """A tuple of action names.

        # Returns

        Tuple of (ordered) action names so that taking action
            running `task.step(i)` corresponds to taking action task.action_names()[i].
        """
        raise NotImplementedError()

    @property
    def total_actions(self) -> int:
        """Total number of actions available to an agent in this Task."""
        return len(self.action_names())

    def index_to_action(self, index: int) -> str:
        """Returns the action name correspond to `index`."""
        assert 0 <= index < self.total_actions
        return self.action_names()[index]

    @abstractmethod
    def close(self) -> None:
        """Closes the environment and any other files opened by the Task (if
        applicable)."""
        raise NotImplementedError()

    def metrics(self) -> Dict[str, Any]:
        """Computes metrics related to the task after the task's completion.

        By default this function is automatically called during training
        and the reported metrics logged to tensorboard.

        # Returns

        A dictionary where every key is a string (the metric's
            name) and the value is the value of the metric.
        """
        return {
            "ep_length": self.num_steps_taken(),
            "reward": self._total_reward,
            "task_info": self.task_info,
        }

    def query_expert(self) -> Tuple[Any, bool]:
        """Query the expert policy for this task.

        # Returns

         A tuple (x, y) where x is the expert action (or policy) and y is False \
            if the expert could not determine the optimal action (otherwise True). Here y \
            is used for masking. Even when y is False, x should still lie in the space of \
            possible values (e.g. if x is the expert policy then x should be the correct length, \
            sum to 1, and have non-negative entries).
        """
        raise NotImplementedError()


SubTaskType = TypeVar("SubTaskType", bound=Task)


class TaskSampler(abc.ABC):
    """Abstract class defining a how new tasks are sampled."""

    @property
    @abstractmethod
    def __len__(self) -> Union[int, float]:
        """Length.

        # Returns

        Number of total tasks remaining that can be sampled. Can be
            float('inf').
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def total_unique(self) -> Optional[Union[int, float]]:
        """Total unique tasks.

        # Returns

        Total number of *unique* tasks that can be sampled. Can be
            float('inf') or, if the total unique is not known, None.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def last_sampled_task(self) -> Optional[Task]:
        """Get the most recently sampled Task.

        # Returns

        The most recently sampled Task.
        """
        raise NotImplementedError()

    @abstractmethod
    def next_task(self, force_advance_scene: bool = False) -> Optional[Task]:
        """Get the next task in the sampler's stream.

        # Parameters

        force_advance_scene : Used to (if applicable) force the task sampler to
            use a new scene for the next task. This is useful if, during training,
            you would like to train with one scene for some number of steps and
            then explicitly control when you begin training with the next scene.

        # Returns

        The next Task in the sampler's stream if a next task exists. Otherwise None.
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
        """Checks if all observation spaces of tasks that can be sampled are
        equal.

        # Returns

        True if all Tasks that can be sampled by this sampler have the
            same observation space. Otherwise False.
        """
        raise NotImplementedError()

    @abstractmethod
    def reset(self) -> None:
        """Resets task sampler to its original state (except for any seed).
        """
        raise NotImplementedError()

    @abstractmethod
    def set_seed(self, seed: int) -> None:
        """Sets new RNG seed.

        # Parameters

        seed : New seed.
        """
        raise NotImplementedError()
