import random
from typing import Any, List, Dict, Optional, Union, Callable, Sequence, Tuple

import gym
import numpy as np
from gym.utils import seeding

from allenact.base_abstractions.misc import RLStepResult
from allenact.base_abstractions.sensor import Sensor, SensorSuite
from allenact.base_abstractions.task import Task, TaskSampler
from allenact.utils.experiment_utils import set_seed
from allenact.utils.system import get_logger
from allenact_plugins.gym_plugin.gym_environment import GymEnvironment
from allenact_plugins.gym_plugin.gym_sensors import GymBox2DSensor, GymMuJoCoSensor


class GymTask(Task[gym.Env]):
    """Abstract gym task.

    Subclasses need to implement `class_action_names` and `_step`.
    """

    def __init__(
        self,
        env: GymEnvironment,
        sensors: Union[SensorSuite, List[Sensor]],
        task_info: Dict[str, Any],
        **kwargs,
    ):
        max_steps = env.spec.max_episode_steps
        super().__init__(
            env=env, sensors=sensors, task_info=task_info, max_steps=max_steps, **kwargs
        )
        self._gym_done = False
        self.task_name: str = self.env.spec.id

    @property
    def action_space(self) -> gym.spaces.Space:
        return self.env.action_space

    def render(self, mode: str = "rgb", *args, **kwargs) -> np.ndarray:
        if mode == "rgb":
            mode = "rgb_array"
        return self.env.render(mode=mode)

    def get_observations(
        self, *args, gym_obs: Optional[Dict[str, Any]] = None, **kwargs
    ) -> Any:
        return self.sensor_suite.get_observations(
            env=self.env, task=self, gym_obs=gym_obs
        )

    def reached_terminal_state(self) -> bool:
        return self._gym_done

    def close(self) -> None:
        pass

    def metrics(self) -> Dict[str, Any]:
        # noinspection PyUnresolvedReferences,PyCallingNonCallable
        env_metrics = self.env.metrics() if hasattr(self.env, "metrics") else {}
        return {
            **super().metrics(),
            **{k: float(v) for k, v in env_metrics.items()},
            "success": int(
                self.env.was_successful
                if hasattr(self.env, "was_successful")
                else self.cumulative_reward > 0
            ),
        }


class GymContinuousTask(GymTask):
    """Task for a continuous-control gym Box2D & MuJoCo Env; it allows
    interfacing allenact with gym tasks."""

    @classmethod
    def class_action_names(cls, **kwargs) -> Tuple[str, ...]:
        return tuple()

    def _step(self, action: Sequence[float]) -> RLStepResult:
        action = np.array(action)

        gym_obs, reward, self._gym_done, info = self.env.step(action=action)

        return RLStepResult(
            observation=self.get_observations(gym_obs=gym_obs),
            reward=reward,
            done=self.is_done(),
            info=info,
        )


def default_task_selector(env_name: str) -> type:
    """Helper function for `GymTaskSampler`."""
    if env_name in [
        # Box2d Env
        "CarRacing-v0",
        "LunarLanderContinuous-v2",
        "BipedalWalker-v2",
        "BipedalWalkerHardcore-v2",
        # MuJoCo Env
        "InvertedPendulum-v2",
        "Ant-v2",
        "InvertedDoublePendulum-v2",
        "Humanoid-v2",
        "Reacher-v2",
        "Hopper-v2",
        "HalfCheetah-v2",
        "Swimmer-v2",
        "Walker2d-v2",
    ]:
        return GymContinuousTask
    raise NotImplementedError()


def sensor_selector(env_name: str) -> Sensor:
    """Helper function for `GymTaskSampler`."""
    if env_name in [
        "CarRacing-v0",
        "LunarLanderContinuous-v2",
        "BipedalWalker-v2",
        "BipedalWalkerHardcore-v2",
        "LunarLander-v2",
    ]:
        return GymBox2DSensor(env_name)
    elif env_name in [
        "InvertedPendulum-v2",
        "Ant-v2",
        "InvertedDoublePendulum-v2",
        "Humanoid-v2",
        "Reacher-v2",
        "Hopper-v2",
        "HalfCheetah-v2",
        "Swimmer-v2",
        "Walker2d-v2",
    ]:
        return GymMuJoCoSensor(gym_env_name=env_name, uuid="gym_mujoco_data")
    raise NotImplementedError()


class GymTaskSampler(TaskSampler):
    """TaskSampler for gym environments."""

    def __init__(
        self,
        gym_env_type: str = "LunarLanderContinuous-v2",
        sensors: Optional[Union[SensorSuite, List[Sensor]]] = None,
        max_tasks: Optional[int] = None,
        num_unique_seeds: Optional[int] = None,
        task_seeds_list: Optional[List[int]] = None,
        deterministic_sampling: bool = False,
        task_selector: Callable[[str], type] = default_task_selector,
        repeat_failed_task_for_min_steps: int = 0,
        extra_task_kwargs: Optional[Dict] = None,
        seed: Optional[int] = None,
        **kwargs,
    ):
        super().__init__()

        self.gym_env_type = gym_env_type

        self.sensors: SensorSuite
        if sensors is None:
            self.sensors = SensorSuite([sensor_selector(self.gym_env_type)])
        else:
            self.sensors = (
                SensorSuite(sensors)
                if not isinstance(sensors, SensorSuite)
                else sensors
            )

        self.max_tasks = max_tasks
        self.num_unique_seeds = num_unique_seeds
        self.deterministic_sampling = deterministic_sampling
        self.repeat_failed_task_for_min_steps = repeat_failed_task_for_min_steps
        self.extra_task_kwargs = (
            extra_task_kwargs if extra_task_kwargs is not None else {}
        )

        self._last_env_seed: Optional[int] = None
        self._last_task: Optional[GymTask] = None
        self._number_of_steps_taken_with_task_seed = 0

        assert (not deterministic_sampling) or repeat_failed_task_for_min_steps <= 0, (
            "If `deterministic_sampling` is True then we require"
            " `repeat_failed_task_for_min_steps <= 0`"
        )
        assert (self.num_unique_seeds is None) or (
            0 < self.num_unique_seeds
        ), "`num_unique_seeds` must be a positive integer."

        self.num_unique_seeds = num_unique_seeds
        self.task_seeds_list = task_seeds_list
        if self.task_seeds_list is not None:
            if self.num_unique_seeds is not None:
                assert self.num_unique_seeds == len(
                    self.task_seeds_list
                ), "`num_unique_seeds` must equal the length of `task_seeds_list` if both specified."
            self.num_unique_seeds = len(self.task_seeds_list)
        elif self.num_unique_seeds is not None:
            self.task_seeds_list = list(range(self.num_unique_seeds))
        if num_unique_seeds is not None and repeat_failed_task_for_min_steps > 0:
            raise NotImplementedError(
                "`repeat_failed_task_for_min_steps` must be <=0 if number"
                " of unique seeds is not None."
            )

        assert (not deterministic_sampling) or (
            self.num_unique_seeds is not None
        ), "Cannot use deterministic sampling when `num_unique_seeds` is `None`."

        if (not deterministic_sampling) and self.max_tasks:
            get_logger().warning(
                "`deterministic_sampling` is `False` but you have specified `max_tasks < inf`,"
                " this might be a mistake when running testing."
            )

        if seed is not None:
            self.set_seed(seed)
        else:
            self.np_seeded_random_gen, _ = seeding.np_random(
                random.randint(0, 2 ** 31 - 1)
            )

        self.num_tasks_generated = 0
        self.task_type = task_selector(self.gym_env_type)
        self.env: GymEnvironment = GymEnvironment(self.gym_env_type)

    @property
    def length(self) -> Union[int, float]:
        return (
            float("inf")
            if self.max_tasks is None
            else self.max_tasks - self.num_tasks_generated
        )

    @property
    def total_unique(self) -> Optional[Union[int, float]]:
        return None if self.num_unique_seeds is None else self.num_unique_seeds

    @property
    def last_sampled_task(self) -> Optional[Task]:
        raise NotImplementedError

    def next_task(self, force_advance_scene: bool = False) -> Optional[GymTask]:
        if self.length <= 0:
            return None

        repeating = False
        if self.num_unique_seeds is not None:
            if self.deterministic_sampling:
                self._last_env_seed = self.task_seeds_list[
                    self.num_tasks_generated % len(self.task_seeds_list)
                ]
            else:
                self._last_env_seed = self.np_seeded_random_gen.choice(
                    self.task_seeds_list
                )
        else:
            if self._last_task is not None:
                self._number_of_steps_taken_with_task_seed += (
                    self._last_task.num_steps_taken()
                )

            if (
                self._last_env_seed is not None
                and self._number_of_steps_taken_with_task_seed
                < self.repeat_failed_task_for_min_steps
                and self._last_task.cumulative_reward == 0
            ):
                repeating = True
            else:
                self._number_of_steps_taken_with_task_seed = 0
                self._last_env_seed = self.np_seeded_random_gen.randint(0, 2 ** 31 - 1)

        task_has_same_seed_reset = hasattr(self.env, "same_seed_reset")

        if repeating and task_has_same_seed_reset:
            # noinspection PyUnresolvedReferences
            self.env.same_seed_reset()
        else:
            self.env.seed(self._last_env_seed)
            self.env.saved_seed = self._last_env_seed
            self.env.reset()

        self.num_tasks_generated += 1

        task_info = {"id": "random%d" % random.randint(0, 2 ** 63 - 1)}

        self._last_task = self.task_type(
            **dict(env=self.env, sensors=self.sensors, task_info=task_info),
            **self.extra_task_kwargs,
        )

        return self._last_task

    def close(self) -> None:
        self.env.close()

    @property
    def all_observation_spaces_equal(self) -> bool:
        return True

    def reset(self) -> None:
        self.num_tasks_generated = 0
        self.env.reset()

    def set_seed(self, seed: int) -> None:
        self.np_seeded_random_gen, _ = seeding.np_random(seed)
        if seed is not None:
            set_seed(seed)
