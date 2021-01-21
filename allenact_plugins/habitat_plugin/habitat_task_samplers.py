from typing import List, Optional, Union, Callable

import gym
import habitat
from habitat.config import Config

from allenact.base_abstractions.sensor import Sensor
from allenact.base_abstractions.task import TaskSampler
from allenact_plugins.habitat_plugin.habitat_environment import HabitatEnvironment
from allenact_plugins.habitat_plugin.habitat_tasks import PointNavTask, ObjectNavTask  # type: ignore


class PointNavTaskSampler(TaskSampler):
    def __init__(
        self,
        env_config: Config,
        sensors: List[Sensor],
        max_steps: int,
        action_space: gym.Space,
        distance_to_goal: float,
        filter_dataset_func: Optional[
            Callable[[habitat.Dataset], habitat.Dataset]
        ] = None,
        **task_init_kwargs,
    ) -> None:
        self.grid_size = 0.25
        self.env: Optional[HabitatEnvironment] = None
        self.max_tasks: Optional[int] = None
        self.reset_tasks: Optional[int] = None
        self.sensors = sensors
        self.max_steps = max_steps
        self._action_space = action_space
        self.env_config = env_config
        self.distance_to_goal = distance_to_goal
        self.seed: Optional[int] = None
        self.filter_dataset_func = filter_dataset_func

        self._last_sampled_task: Optional[PointNavTask] = None

        self.task_init_kwargs = task_init_kwargs

    def _create_environment(self) -> HabitatEnvironment:
        dataset = habitat.make_dataset(
            self.env_config.DATASET.TYPE, config=self.env_config.DATASET
        )
        if len(dataset.episodes) == 0:
            raise RuntimeError("Empty input dataset.")

        if self.filter_dataset_func is not None:
            dataset = self.filter_dataset_func(dataset)
            if len(dataset.episodes) == 0:
                raise RuntimeError("Empty dataset after filtering.")

        env = HabitatEnvironment(config=self.env_config, dataset=dataset)
        self.max_tasks = (
            None if self.env_config.MODE == "train" else env.num_episodes
        )  # env.num_episodes
        self.reset_tasks = self.max_tasks
        return env

    @property
    def length(self) -> Union[int, float]:
        """
        @return: Number of total tasks remaining that can be sampled. Can be float('inf').
        """
        return float("inf") if self.max_tasks is None else self.max_tasks

    @property
    def total_unique(self) -> Union[int, float, None]:
        return self.env.num_episodes

    @property
    def last_sampled_task(self) -> Optional[PointNavTask]:
        return self._last_sampled_task

    def close(self) -> None:
        if self.env is not None:
            self.env.stop()

    @property
    def all_observation_spaces_equal(self) -> bool:
        """
        @return: True if all Tasks that can be sampled by this sampler have the
            same observation space. Otherwise False.
        """
        return True

    def next_task(self, force_advance_scene=False) -> Optional[PointNavTask]:
        if self.max_tasks is not None and self.max_tasks <= 0:
            return None

        if self.env is not None:
            self.env.reset()
        else:
            self.env = self._create_environment()
            self.env.reset()
        ep_info = self.env.get_current_episode()
        target = ep_info.goals[0].position

        task_info = {
            "target": target,
            "distance_to_goal": self.distance_to_goal,
        }

        self._last_sampled_task = PointNavTask(
            env=self.env,
            sensors=self.sensors,
            task_info=task_info,
            max_steps=self.max_steps,
            action_space=self._action_space,
            **self.task_init_kwargs,
        )

        if self.max_tasks is not None:
            self.max_tasks -= 1

        return self._last_sampled_task

    def reset(self):
        self.max_tasks = self.reset_tasks

    def set_seed(self, seed: int):
        self.seed = seed
        if seed is not None:
            self.env.env.seed(seed)


class ObjectNavTaskSampler(TaskSampler):
    def __init__(
        self,
        env_config: Config,
        sensors: List[Sensor],
        max_steps: int,
        action_space: gym.Space,
        distance_to_goal: float,
        **kwargs,
    ) -> None:
        self.grid_size = 0.25
        self.env: Optional[HabitatEnvironment] = None
        self.max_tasks: Optional[int] = None
        self.reset_tasks: Optional[int] = None
        self.sensors = sensors
        self.max_steps = max_steps
        self._action_space = action_space
        self.env_config = env_config
        self.distance_to_goal = distance_to_goal
        self.seed: Optional[int] = None

        self._last_sampled_task: Optional[ObjectNavTask] = None

    def _create_environment(self) -> HabitatEnvironment:
        dataset = habitat.make_dataset(
            self.env_config.DATASET.TYPE, config=self.env_config.DATASET
        )
        env = HabitatEnvironment(config=self.env_config, dataset=dataset)
        self.max_tasks = (
            None if self.env_config.MODE == "train" else env.num_episodes
        )  # mp3d objectnav val -> 2184
        self.reset_tasks = self.max_tasks
        return env

    @property
    def length(self) -> Union[int, float]:
        """
        @return: Number of total tasks remaining that can be sampled. Can be float('inf').
        """
        return float("inf") if self.max_tasks is None else self.max_tasks

    @property
    def total_unique(self) -> Union[int, float, None]:
        return self.env.num_episodes

    @property
    def last_sampled_task(self) -> Optional[ObjectNavTask]:
        return self._last_sampled_task

    def close(self) -> None:
        if self.env is not None:
            self.env.stop()

    @property
    def all_observation_spaces_equal(self) -> bool:
        """
        @return: True if all Tasks that can be sampled by this sampler have the
            same observation space. Otherwise False.
        """
        return True

    def next_task(self, force_advance_scene=False) -> Optional[ObjectNavTask]:
        if self.max_tasks is not None and self.max_tasks <= 0:
            return None

        if self.env is not None:
            self.env.reset()
        else:
            self.env = self._create_environment()
            self.env.reset()
        ep_info = self.env.get_current_episode()
        target = ep_info.goals[0].position

        task_info = {
            "target": target,
            "distance_to_goal": self.distance_to_goal,
        }

        self._last_sampled_task = ObjectNavTask(
            env=self.env,
            sensors=self.sensors,
            task_info=task_info,
            max_steps=self.max_steps,
            action_space=self._action_space,
        )

        if self.max_tasks is not None:
            self.max_tasks -= 1

        return self._last_sampled_task

    def reset(self):
        self.max_tasks = self.reset_tasks

    def set_seed(self, seed: int):
        self.seed = seed
        if seed is not None:
            self.env.env.seed(seed)
