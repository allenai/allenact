from typing import Tuple, List, Dict, Any, Optional

import gym
import numpy as np

from rl_habitat.habitat_constants import (
    MOVE_AHEAD,
    ROTATE_LEFT,
    ROTATE_RIGHT,
    END,
    LOOK_UP,
    LOOK_DOWN,
)
from rl_habitat.habitat_environment import HabitatEnvironment
from rl_base.common import RLStepResult
from rl_base.sensor import Sensor
from rl_base.task import Task


class HabitatTask(Task[HabitatEnvironment]):
    def __init__(
        self,
        env: HabitatEnvironment,
        sensors: List[Sensor],
        task_info: Dict[str, Any],
        max_steps: int,
        **kwargs
    ) -> None:
        super().__init__(
            env=env, sensors=sensors, task_info=task_info, max_steps=max_steps, **kwargs
        )

        self._last_action: Optional[str] = None
        self._last_action_ind: Optional[int] = None
        self._last_action_success: Optional[bool] = None

    @property
    def last_action(self):
        return self._last_action

    @last_action.setter
    def last_action(self, value: str):
        self._last_action = value

    @property
    def last_action_success(self):
        return self._last_action_success

    @last_action_success.setter
    def last_action_success(self, value: Optional[bool]):
        self._last_action_success = value

    def _step(self, action: int) -> RLStepResult:
        self._last_action_ind = action
        self.last_action = self.action_names()[action]
        self.last_action_success = None
        step_result = super(HabitatTask, self).step(action=action)
        step_result.info["action"] = self._last_action_ind
        step_result.info["action_success"] = self.last_action_success
        return step_result

    def render(self, mode: str = "rgb", *args, **kwargs) -> np.ndarray:
        if mode == "rgb":
            return self.env.current_frame["rgb"]
        elif mode == "depth":
            return self.env.current_frame["depth"]
        else:
            raise NotImplementedError()


class PointNavTask(Task[HabitatTask]):
    _actions = (MOVE_AHEAD, ROTATE_LEFT, ROTATE_RIGHT, END)

    def __init__(
        self,
        env: HabitatEnvironment,
        sensors: List[Sensor],
        task_info: Dict[str, Any],
        max_steps: int,
        **kwargs
    ) -> None:
        # print("task info in objectnavtask %s" % task_info)
        super().__init__(
            env=env, sensors=sensors, task_info=task_info, max_steps=max_steps, **kwargs
        )
        self._took_end_action: bool = False
        self._success: Optional[bool] = False
        self._subsampled_locations_from_which_obj_visible = None
        self.last_geodesic_distance = self.env.env.get_metrics()['distance_to_goal']
        self._rewards = []
        self._distance_to_goal = []
        self._metrics = None

    @property
    def action_space(self):
        return gym.spaces.Discrete(len(self._actions))

    def reached_terminal_state(self) -> bool:
        return self.env.env.episode_over

    @classmethod
    def action_names(cls) -> Tuple[str, ...]:
        return cls._actions

    def close(self) -> None:
        self.env.stop()

    def _step(self, action: int) -> RLStepResult:
        action_str = self.action_names()[action]

        self.env.step({"action": action_str})

        if action_str == END:
            self._took_end_action = True
            self._success = self._is_goal_in_range()
            self.last_action_success = self._success
        else:
            self.last_action_success = self.env.last_action_success

        step_result = RLStepResult(
            observation=self.get_observations(),
            reward=self.judge(),
            done=self.is_done(),
            info={"last_action_success": self.last_action_success},
        )
        return step_result

    def render(self, mode: str = "rgb", *args, **kwargs) -> np.ndarray:
        assert mode in ["rgb", "depth"], "only rgb and depth rendering is implemented"
        return self.env.current_frame['rgb']

    def _is_goal_in_range(self) -> bool:
        # The habitat simulator will return an SPL value of 0.0 whenever the goal is not in range
        return bool(self.env.env.get_metrics()['spl'])

    def judge(self) -> float:
        reward = -0.01

        geodesic_distance = self.env.env.get_metrics()['distance_to_goal']
        delta_distance_reward = self.last_geodesic_distance - geodesic_distance
        reward += delta_distance_reward
        self.last_geodesic_distance = geodesic_distance

        if self._took_end_action:
            reward += 10.0 if self._success else 0.0

        self._rewards.append(float(reward))

        return float(reward)

    def metrics(self) -> Dict[str, Any]:
        if not self.is_done():
            return {}
        else:
            _metrics = self.env.env.get_metrics()
            metrics = {
                "success": self._success,
                "ep_length": self.num_steps_taken(),
                "total_reward": np.sum(self._rewards),
                "spl": _metrics['spl'] if _metrics['spl'] is not None else 0.0
            }
            self._rewards = []
            return metrics

    def query_expert(self) -> Tuple[int, bool]:
        if self._is_goal_in_range():
            return self.action_names().index(END), True

        target = self.task_info["target"]
        current_location = self.env.get_location()
        path = self.env.get_shortest_path(
            source_state=current_location, goal_state=target
        )
        # TODO convert returned path to optimal action
        print("path:", path)
        exit()
        return path[0], True


class ObjectNavTask(Task[HabitatTask]):
    _actions = (MOVE_AHEAD, ROTATE_LEFT, ROTATE_RIGHT, END, LOOK_UP, LOOK_DOWN)

    def __init__(
        self,
        env: HabitatEnvironment,
        sensors: List[Sensor],
        task_info: Dict[str, Any],
        max_steps: int,
        **kwargs
    ) -> None:
        # print("task info in objectnavtask %s" % task_info)
        super().__init__(
            env=env, sensors=sensors, task_info=task_info, max_steps=max_steps, **kwargs
        )
        self._took_end_action: bool = False
        self._success: Optional[bool] = False
        self._subsampled_locations_from_which_obj_visible = None

        # self.last_geodesic_distance = env.get_geodesic_distance() # self.env.get_current_episode().info['geodesic_distance']
        # self.last_distance_to_goal = self.env.get_current_episode().info['geodesic_distance'] #self.env.env.get_metrics()["distance_to_goal"]

        self.last_geodesic_distance = self.env.env.get_metrics()['distance_to_goal']

        self._rewards = []
        self._distance_to_goal = []
        self._metrics = None

    @property
    def action_space(self):
        return gym.spaces.Discrete(len(self._actions))

    def reached_terminal_state(self) -> bool:
        return self.env.env.episode_over

    @classmethod
    def action_names(cls) -> Tuple[str, ...]:
        return cls._actions

    def close(self) -> None:
        self.env.stop()

    def _step(self, action: int) -> RLStepResult:
        action_str = self.action_names()[action]

        self.env.step({"action": action_str})

        if action_str == END:
            self._took_end_action = True
            self._success = self._is_goal_in_range()
            self.last_action_success = self._success
        else:
            self.last_action_success = self.env.last_action_success

        step_result = RLStepResult(
            observation=self.get_observations(),
            reward=self.judge(),
            done=self.is_done(),
            info={"last_action_success": self.last_action_success},
        )
        return step_result

    def render(self, mode: str = "rgb", *args, **kwargs) -> np.ndarray:
        assert mode in ["rgb", "depth"], "only rgb and depth rendering is implemented"
        return self.env.current_frame['rgb']

    def _is_goal_in_range(self) -> bool:
        # The habitat simulator will return an SPL value of 0.0 whenever the goal is not in range
        return bool(self.env.env.get_metrics()['spl'])

    def judge(self) -> float:
        reward = -0.01

        # last_geodesic_distance = self.env.last_geodesic_distance
        # new_geodesic_distance = self.env.get_geodesic_distance()
        new_geodesic_distance = self.env.env.get_metrics()['distance_to_goal']
        delta_distance_reward = self.last_geodesic_distance - new_geodesic_distance

        reward += delta_distance_reward

        if self._took_end_action:
            reward += 10.0 if self._success else 0.0

        self._rewards.append(float(reward))
        self.last_geodesic_distance = new_geodesic_distance

        return float(reward)

    def metrics(self) -> Dict[str, Any]:
        # print("Self Rewards:", self._rewards)
        if not self.is_done():
            return {}
        else:
            _metrics = self.env.env.get_metrics()
            metrics = {
                "success": self._success,
                "ep_length": self.num_steps_taken(),
                "total_reward": np.sum(self._rewards),
                "spl": _metrics['spl'] if _metrics['spl'] is not None else 0.0
            }
            self._rewards = []
            return metrics

    def query_expert(self) -> Tuple[int, bool]:
        if self._is_goal_in_range():
            return self.action_names().index(END), True

        target = self.task_info["target"]
        current_location = self.env.get_location()
        path = self.env.get_shortest_path(
            source_state=current_location, goal_state=target
        )
        # TODO convert returned path to optimal action
        print("path:", path)
        exit()
        return path[0], True
