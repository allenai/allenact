from typing import List, Optional, Union

from rl_habitat.habitat_environment import HabitatEnvironment
import gym
import habitat
from habitat.config import Config

from rl_base.sensor import Sensor
from rl_base.task import TaskSampler
from rl_habitat.habitat_tasks import PointNavTask, ObjectNavTask


class PointNavTaskSampler(TaskSampler):
    def __init__(
        self,
        env_config: Config,
        sensors: List[Sensor],
        max_steps: int,
        action_space: gym.Space,
        distance_to_goal: float,
        *args,
        **kwargs
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

        self._last_sampled_task: Optional[PointNavTask] = None

    def _create_environment(self) -> HabitatEnvironment:
        dataset = habitat.make_dataset(
            self.env_config.DATASET.TYPE, config=self.env_config.DATASET
        )
        env = HabitatEnvironment(
            config=self.env_config,
            dataset=dataset
        )
        self.max_tasks =  None if self.env_config.MODE == 'train' else env.num_episodes  # env.num_episodes
        self.reset_tasks = self.max_tasks
        return env

    @property
    def __len__(self) -> Union[int, float]:
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

    def next_task(self, force_advance_scene=False) -> PointNavTask:
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
            "actions": []
        }

        self._last_sampled_task = PointNavTask(
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
        # if seed is not None:
        #     set_seed(seed)


class ObjectNavTaskSampler(TaskSampler):
    def __init__(
        self,
        env_config: Config,
        sensors: List[Sensor],
        max_steps: int,
        action_space: gym.Space,
        distance_to_goal: float,
        *args,
        **kwargs
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

        self._last_sampled_task: Optional[PointNavTask] = None

    def _create_environment(self) -> HabitatEnvironment:
        dataset = habitat.make_dataset(
            self.env_config.DATASET.TYPE, config=self.env_config.DATASET
        )
        env = HabitatEnvironment(
            config=self.env_config,
            dataset=dataset
        )
        self.max_tasks = None if self.env_config.MODE == 'train' else env.num_episodes  # mp3d objectnav val -> 2184
        self.reset_tasks = self.max_tasks
        return env

    @property
    def __len__(self) -> Union[int, float]:
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

    def next_task(self, force_advance_scene=False) -> ObjectNavTask:
        if self.max_tasks is not None and self.max_tasks <= 0:
            return None

        if self.env is not None:
            self.env.reset()
        else:
            self.env = self._create_environment()
            self.env.reset()
        ep_info = self.env.get_current_episode()
        while ep_info.goals[0].object_category != 'chair':
            self.env.reset()
            ep_info = self.env.get_current_episode()
            if self.max_tasks is not None:
                self.max_tasks -= 1
        target = ep_info.goals[0].position

        task_info = {
            "target": target,
            "distance_to_goal": self.distance_to_goal,
            "actions": []
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
        # if seed is not None:
        #     set_seed(seed)
