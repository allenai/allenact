from typing import Tuple, List, Dict, Any, Optional

import gym
import numpy as np

from extensions.habitat.constants import (
    MOVE_AHEAD,
    ROTATE_LEFT,
    ROTATE_RIGHT,
    END,
)
from extensions.habitat.environment import HabitatEnvironment
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

    @property
    def action_space(self):
        return gym.spaces.Discrete(len(self._actions))

    def reached_terminal_state(self) -> bool:
        return self._took_end_action

    @classmethod
    def action_names(cls) -> Tuple[str, ...]:
        return cls._actions

    def close(self) -> None:
        self.env.stop()

    def _step(self, action: int) -> RLStepResult:
        action_str = self.action_names()[action]

        if action_str == END:
            self._took_end_action = True
            self._success = self._is_goal_in_range()
            self.last_action_success = self._success
        else:
            self.env.step({"action": action_str})
            self.last_action_success = self.env.last_action_success

            # if (
            #     not self.last_action_success
            # ) and self._CACHED_LOCATIONS_FROM_WHICH_OBJECT_IS_VISIBLE is not None:
            #     self.env.update_graph_with_failed_action(failed_action=action_str)

        step_result = RLStepResult(
            observation=self.get_observations(),
            reward=self.judge(),
            done=self.is_done(),
            info={"last_action_success": self.last_action_success},
        )
        return step_result

    def render(self, mode: str = "rgb", *args, **kwargs) -> np.ndarray:
        assert mode in ["rgb", "depth"], "only rgb and depth rendering is implemented"
        return self.env.current_frame

    def _is_goal_in_range(self) -> bool:
        # target = self.task_info["target"]
        # current_location = self.env.get_location()
        # distance = self.env.get_geodesic_distance(
        #     source_state=current_location, goal_state=target
        # )
        geodesic_distance = self.env.get_current_episode().info['geodesic_distance']
        # TODO: Plug in actual distance here!
        return geodesic_distance < 1.0

    def judge(self) -> float:
        reward = -0.01

        if not self.last_action_success:
            reward += -0.1

        if self._took_end_action:
            reward += 1.0 if self._success else -1.0

        return float(reward)

    def metrics(self) -> Dict[str, Any]:
        if not self.is_done():
            return {}
        else:
            return {"success": self._success, "ep_length": self.num_steps_taken()}

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
