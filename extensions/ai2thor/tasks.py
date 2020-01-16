from typing import Tuple, List, Dict, Any, Optional

import numpy as np

from extensions.ai2thor.constants import (
    MOVE_AHEAD,
    ROTATE_LEFT,
    ROTATE_RIGHT,
    LOOK_DOWN,
    LOOK_UP,
)
from extensions.ai2thor.environment import AI2ThorEnvironment
from rl_base.common import RLStepResult
from rl_base.sensor import Sensor
from rl_base.task import Task


class AI2ThorTask(Task[AI2ThorEnvironment]):
    def __init__(
        self,
        env: AI2ThorEnvironment,
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

    def step(self, action: int) -> Dict[str, Any]:
        self._last_action_ind = action
        self.last_action = self.action_names()[action]
        self.last_action_success = None
        step_result = super(AI2ThorTask, self).step(action=action)
        step_result.info["action"] = self._last_action_ind
        step_result.info["action_success"] = self.last_action_success
        return step_result

    def render(self, mode: str = "rgb", *args, **kwargs) -> np.ndarray:
        if mode == "rgb":
            return self.env.current_frame
        else:
            raise NotImplementedError()


class ObjectNavTask(Task[AI2ThorEnvironment]):
    _actions = (MOVE_AHEAD, ROTATE_LEFT, ROTATE_RIGHT, LOOK_DOWN, LOOK_UP)

    @property
    def action_space(self):
        raise NotImplementedError()

    def reached_terminal_state(self) -> bool:
        return self._is_goal_object_visible()

    @classmethod
    def action_names(cls) -> Tuple[str, ...]:
        return cls._actions

    def close(self) -> None:
        raise NotImplementedError()

    def _step(self, action: int) -> RLStepResult:
        action_str = self.action_names()[action]
        self.env.step({"action": action_str})
        self.last_action_success = self.env.last_action_success

        step_result = RLStepResult(
            observation=self.get_observations(),
            reward=self.judge(),
            done=self.is_done(),
            info={"last_action_success": self.last_action_success},
        )
        return step_result

    def _is_goal_object_visible(self) -> bool:
        return any(
            o["objectType"] == self.task_info["object_type"]
            for o in self.env.visible_objects()
        )

    def judge(self) -> float:
        reward = -0.01

        if not self.last_action_success:
            reward += -0.1
        elif self._is_goal_object_visible():
            reward += 1.0
        return float(reward)
