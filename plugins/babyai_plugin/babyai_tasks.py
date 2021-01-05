import random
import signal
from typing import Tuple, Any, List, Dict, Optional, Union, Callable, Sequence, cast

import babyai
import babyai.bot
import gym
import numpy as np
from gym.utils import seeding
from gym_minigrid.minigrid import MiniGridEnv
import torch

from core.base_abstractions.misc import RLStepResult
from core.base_abstractions.sensor import Sensor, SensorSuite
from core.base_abstractions.task import Task, TaskSampler
from utils.system import get_logger


class BabyAITask(Task[MiniGridEnv]):
    def __init__(
        self,
        env: MiniGridEnv,
        sensors: Union[SensorSuite, List[Sensor]],
        task_info: Dict[str, Any],
        expert_view_size: int = 7,
        expert_can_see_through_walls: bool = False,
        **kwargs,
    ):
        super().__init__(
            env=env,
            sensors=sensors,
            task_info=task_info,
            max_steps=env.max_steps,
            **kwargs,
        )
        self._was_successful: bool = False
        self.bot: Optional[babyai.bot.Bot] = None
        self._bot_died = False
        self.expert_view_size = expert_view_size
        self.expert_can_see_through_walls = expert_can_see_through_walls
        self._last_action: Optional[int] = None

        env.max_steps = env.max_steps + 1

    @property
    def action_space(self) -> gym.spaces.Discrete:
        return self.env.action_space

    def render(self, mode: str = "rgb", *args, **kwargs) -> np.ndarray:
        return self.env.render(mode=mode)

    def _step(self, action: torch.Tensor) -> RLStepResult:
        # assert isinstance(action, int)
        # action = cast(int, action)
        action = action.item()

        minigrid_obs, reward, done, info = self.env.step(action=action)
        self._last_action = action

        self._was_successful = done and reward > 0

        return RLStepResult(
            observation=self.get_observations(minigrid_output_obs=minigrid_obs),
            reward=reward,
            done=self.is_done(),
            info=info,
        )

    def get_observations(
        self, *args, minigrid_output_obs: Optional[Dict[str, Any]] = None, **kwargs
    ) -> Any:
        return self.sensor_suite.get_observations(
            env=self.env, task=self, minigrid_output_obs=minigrid_output_obs
        )

    def reached_terminal_state(self) -> bool:
        return self._was_successful

    @classmethod
    def class_action_names(cls, **kwargs) -> Tuple[str, ...]:
        return tuple(
            x
            for x, _ in sorted(
                [(str(a), a.value) for a in MiniGridEnv.Actions], key=lambda x: x[1]
            )
        )

    def close(self) -> None:
        pass

    def _expert_timeout_hander(self, signum, frame):
        raise TimeoutError

    def query_expert(self, **kwargs) -> Tuple[Any, bool]:
        see_through_walls = self.env.see_through_walls
        agent_view_size = self.env.agent_view_size

        if self._bot_died:
            return 0, False

        try:
            self.env.agent_view_size = self.expert_view_size
            self.env.expert_can_see_through_walls = self.expert_can_see_through_walls

            if self.bot is None:
                self.bot = babyai.bot.Bot(self.env)

            signal.signal(signal.SIGALRM, self._expert_timeout_hander)
            signal.alarm(kwargs.get("timeout", 4 if self.num_steps_taken() == 0 else 2))
            return self.bot.replan(self._last_action), True
        except TimeoutError as _:
            self._bot_died = True
            return 0, False
        finally:
            signal.alarm(0)
            self.env.see_through_walls = see_through_walls
            self.env.agent_view_size = agent_view_size

    def metrics(self) -> Dict[str, Any]:
        metrics = {
            **super(BabyAITask, self).metrics(),
            "success": 1.0 * (self.reached_terminal_state()),
        }
        return metrics


class BabyAITaskSampler(TaskSampler):
    def __init__(
        self,
        env_builder: Union[str, Callable[..., MiniGridEnv]],
        sensors: Union[SensorSuite, List[Sensor]],
        max_tasks: Optional[int] = None,
        num_unique_seeds: Optional[int] = None,
        task_seeds_list: Optional[List[int]] = None,
        deterministic_sampling: bool = False,
        extra_task_kwargs: Optional[Dict] = None,
        **kwargs,
    ):
        super(BabyAITaskSampler, self).__init__()
        self.sensors = (
            SensorSuite(sensors) if not isinstance(sensors, SensorSuite) else sensors
        )
        self.max_tasks = max_tasks
        self.num_unique_seeds = num_unique_seeds
        self.deterministic_sampling = deterministic_sampling
        self.extra_task_kwargs = (
            extra_task_kwargs if extra_task_kwargs is not None else {}
        )

        self._last_env_seed: Optional[int] = None
        self._last_task: Optional[BabyAITask] = None

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

        if (not deterministic_sampling) and self.max_tasks:
            get_logger().warning(
                "`deterministic_sampling` is `False` but you have specified `max_tasks < inf`,"
                " this might be a mistake when running testing."
            )

        if isinstance(env_builder, str):
            self.env = gym.make(env_builder)
        else:
            self.env = env_builder()

        self.np_seeded_random_gen, _ = seeding.np_random(random.randint(0, 2 ** 31 - 1))
        self.num_tasks_generated = 0

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

    def next_task(self, force_advance_scene: bool = False) -> Optional[BabyAITask]:
        if self.length <= 0:
            return None

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
            self._last_env_seed = self.np_seeded_random_gen.randint(0, 2 ** 31 - 1)

        self.env.seed(self._last_env_seed)
        self.env.saved_seed = self._last_env_seed
        self.env.reset()

        self.num_tasks_generated += 1
        self._last_task = BabyAITask(env=self.env, sensors=self.sensors, task_info={})
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
