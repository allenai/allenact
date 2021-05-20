import abc
import string
from typing import List, Dict, Any, Optional, Tuple, Union, Sequence, cast

import gym
import numpy as np
from gym.utils import seeding

from allenact.base_abstractions.misc import RLStepResult
from allenact.base_abstractions.sensor import Sensor, SensorSuite
from allenact.base_abstractions.task import Task, TaskSampler
from allenact.utils.experiment_utils import set_seed
from allenact.utils.system import get_logger
from allenact_plugins.lighthouse_plugin.lighthouse_environment import (
    LightHouseEnvironment,
)
from allenact_plugins.lighthouse_plugin.lighthouse_sensors import get_corner_observation

DISCOUNT_FACTOR = 0.99
STEP_PENALTY = -0.01
FOUND_TARGET_REWARD = 1.0


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
        sensors: Union[SensorSuite, List[Sensor]],
        task_info: Dict[str, Any],
        max_steps: int,
        **kwargs,
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

    def step(self, action: Union[int, Sequence[int]]) -> RLStepResult:
        assert isinstance(action, int)
        action = cast(int, action)

        self.last_action = action
        return super(LightHouseTask, self).step(action=action)

    def render(self, mode: str = "array", *args, **kwargs) -> np.ndarray:
        if mode == "array":
            return self.env.render(mode, **kwargs)
        elif mode in ["rgb", "rgb_array", "human"]:
            arr = self.env.render("array", **kwargs)
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
        sensors: Union[SensorSuite, List[Sensor]],
        task_info: Dict[str, Any],
        max_steps: int,
        **kwargs,
    ):
        super().__init__(env, sensors, task_info, max_steps, **kwargs)

        self._found_target = False

    @property
    def action_space(self) -> gym.spaces.Discrete:
        return gym.spaces.Discrete(2 * self.env.world_dim)

    def _step(self, action: Union[int, Sequence[int]]) -> RLStepResult:
        assert isinstance(action, int)
        action = cast(int, action)

        self.env.step(action)
        reward = STEP_PENALTY

        if np.all(self.env.current_position == self.env.goal_position):
            self._found_target = True
            reward += FOUND_TARGET_REWARD
        elif self.num_steps_taken() == self.max_steps - 1:
            reward = STEP_PENALTY / (1 - DISCOUNT_FACTOR)

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

    def query_expert(
        self,
        expert_view_radius: int,
        return_policy: bool = False,
        deterministic: bool = False,
        **kwargs,
    ) -> Tuple[Any, bool]:
        view_tuple = get_corner_observation(
            env=self.env, view_radius=expert_view_radius, view_corner_offsets=None,
        )

        goal = self.env.GOAL
        wrong = self.env.WRONG_CORNER

        if self.env.world_dim == 1:
            left_view, right_view, hitting, last_action = view_tuple

            left = 1
            right = 0

            expert_action: Optional[int] = None
            policy: Optional[np.ndarray] = None

            if left_view == goal:
                expert_action = left
            elif right_view == goal:
                expert_action = right
            elif hitting != 2 * self.env.world_dim:
                expert_action = left if last_action == right else right
            elif left_view == wrong:
                expert_action = right
            elif right_view == wrong:
                expert_action = left
            elif last_action == 2 * self.env.world_dim:
                policy = np.array([0.5, 0.5])
            else:
                expert_action = last_action

            if policy is None:
                policy = np.array([expert_action == right, expert_action == left])

        elif self.env.world_dim == 2:

            tl, tr, bl, br, hitting, last_action = view_tuple

            wall = self.env.WALL

            d, r, u, l, none = 0, 1, 2, 3, 4

            if tr == goal:
                if hitting != r:
                    expert_action = r
                else:
                    expert_action = u
            elif br == goal:
                if hitting != d:
                    expert_action = d
                else:
                    expert_action = r
            elif bl == goal:
                if hitting != l:
                    expert_action = l
                else:
                    expert_action = d
            elif tl == goal:
                if hitting != u:
                    expert_action = u
                else:
                    expert_action = l

            elif tr == wrong and not any(x == wrong for x in [br, bl, tl]):
                expert_action = l
            elif br == wrong and not any(x == wrong for x in [bl, tl, tr]):
                expert_action = u
            elif bl == wrong and not any(x == wrong for x in [tl, tr, br]):
                expert_action = r
            elif tl == wrong and not any(x == wrong for x in [tr, br, bl]):
                expert_action = d

            elif all(x == wrong for x in [tr, br]) and not any(
                x == wrong for x in [bl, tl]
            ):
                expert_action = l
            elif all(x == wrong for x in [br, bl]) and not any(
                x == wrong for x in [tl, tr]
            ):
                expert_action = u

            elif all(x == wrong for x in [bl, tl]) and not any(
                x == wrong for x in [tr, br]
            ):
                expert_action = r
            elif all(x == wrong for x in [tl, tr]) and not any(
                x == wrong for x in [br, bl]
            ):
                expert_action = d

            elif hitting != none and tr == br == bl == tl:
                # Only possible if in 0 vis setting
                if tr == self.env.WRONG_CORNER or last_action == hitting:
                    if last_action == r:
                        expert_action = u
                    elif last_action == u:
                        expert_action = l
                    elif last_action == l:
                        expert_action = d
                    elif last_action == d:
                        expert_action = r
                    else:
                        raise NotImplementedError()
                else:
                    expert_action = last_action

            elif last_action == r and tr == wall:
                expert_action = u

            elif last_action == u and tl == wall:
                expert_action = l

            elif last_action == l and bl == wall:
                expert_action = d

            elif last_action == d and br == wall:
                expert_action = r

            elif last_action == none:
                expert_action = r

            else:
                expert_action = last_action

            policy = np.array(
                [
                    expert_action == d,
                    expert_action == r,
                    expert_action == u,
                    expert_action == l,
                ]
            )
        else:
            raise NotImplementedError("Can only query expert for world dims of 1 or 2.")

        if return_policy:
            return policy, True
        elif deterministic:
            return int(np.argmax(policy)), True
        else:
            return (
                int(np.argmax(np.random.multinomial(1, policy / (1.0 * policy.sum())))),
                True,
            )


class FindGoalLightHouseTaskSampler(TaskSampler):
    def __init__(
        self,
        world_dim: int,
        world_radius: int,
        sensors: Union[SensorSuite, List[Sensor]],
        max_steps: int,
        max_tasks: Optional[int] = None,
        num_unique_seeds: Optional[int] = None,
        task_seeds_list: Optional[List[int]] = None,
        deterministic_sampling: bool = False,
        seed: Optional[int] = None,
        **kwargs,
    ):
        self.env = LightHouseEnvironment(world_dim=world_dim, world_radius=world_radius)

        self._last_sampled_task: Optional[FindGoalLightHouseTask] = None
        self.sensors = (
            SensorSuite(sensors) if not isinstance(sensors, SensorSuite) else sensors
        )
        self.max_steps = max_steps
        self.max_tasks = max_tasks
        self.num_tasks_generated = 0
        self.deterministic_sampling = deterministic_sampling

        self.num_unique_seeds = num_unique_seeds
        self.task_seeds_list = task_seeds_list
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

        assert (not deterministic_sampling) or (
            self.num_unique_seeds is not None
        ), "Cannot use deterministic sampling when `num_unique_seeds` is `None`."

        if (not deterministic_sampling) and self.max_tasks:
            get_logger().warning(
                "`deterministic_sampling` is `False` but you have specified `max_tasks < inf`,"
                " this might be a mistake when running testing."
            )

        self.seed: int = int(
            seed if seed is not None else np.random.randint(0, 2 ** 31 - 1)
        )
        self.np_seeded_random_gen: Optional[np.random.RandomState] = None
        self.set_seed(self.seed)

    @property
    def world_dim(self):
        return self.env.world_dim

    @property
    def world_radius(self):
        return self.env.world_radius

    @property
    def length(self) -> Union[int, float]:
        return (
            float("inf")
            if self.max_tasks is None
            else self.max_tasks - self.num_tasks_generated
        )

    @property
    def total_unique(self) -> Optional[Union[int, float]]:
        n = 2 ** self.world_dim
        return n if self.num_unique_seeds is None else min(n, self.num_unique_seeds)

    @property
    def last_sampled_task(self) -> Optional[Task]:
        return self._last_sampled_task

    def next_task(self, force_advance_scene: bool = False) -> Optional[Task]:
        if self.length <= 0:
            return None

        if self.num_unique_seeds is not None:
            if self.deterministic_sampling:
                seed = self.task_seeds_list[
                    self.num_tasks_generated % len(self.task_seeds_list)
                ]
            else:
                seed = self.np_seeded_random_gen.choice(self.task_seeds_list)
        else:
            seed = self.np_seeded_random_gen.randint(0, 2 ** 31 - 1)

        self.num_tasks_generated += 1

        self.env.set_seed(seed)
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
        self.num_tasks_generated = 0
        self.set_seed(seed=self.seed)

    def set_seed(self, seed: int) -> None:
        set_seed(seed)
        self.np_seeded_random_gen, _ = seeding.np_random(seed)
        self.seed = seed
