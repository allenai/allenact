from typing import Dict, List, Any, Tuple

from ..new_robothor_environment import RoboThorEnvironment
from rl_ai2thor.object_nav.tasks import ObjectNavTask as BaseObjectNavTask
from rl_base.sensor import Sensor
from rl_base.common import RLStepResult
from ..robothor_constants import (
    MOVE_AHEAD,
    MOVE_BACK,
    ROTATE_LEFT,
    ROTATE_RIGHT,
    LOOK_DOWN,
    LOOK_UP,
    END,
)


class ObjectNavTask(BaseObjectNavTask):
    _actions = (
        MOVE_AHEAD,
        MOVE_BACK,
        ROTATE_LEFT,
        ROTATE_RIGHT,
        LOOK_DOWN,
        LOOK_UP,
        END,
    )

    def __init__(
        self,
        env: RoboThorEnvironment,
        sensors: List[Sensor],
        task_info: Dict[str, Any],
        max_steps: int,
        reward_configs: Dict[str, Any],
        **kwargs
    ) -> None:
        super().__init__(env, sensors, task_info, max_steps)
        self.reward_configs = reward_configs
        self.is_robot = False

    def judge(self) -> float:
        """ Judge the last event. """
        reward = self.reward_configs["step_penalty"]

        if not self.last_action_success:
            reward += self.reward_configs["unsuccessful_action_penalty"]

        if self._took_end_action:
            reward += (
                self.reward_configs["goal_success_reward"] if self._success else -1.0
            )

        return float(reward)

    def _step(self, action: int) -> RLStepResult:
        action_str = self.action_names()[action]

        if action_str == END:
            self._took_end_action = True
            self._success = self._is_goal_object_visible()
            self.last_action_success = self._success
        else:
            self.env.step({"action": action_str})
            self.last_action_success = self.env.last_action_success

        step_result = RLStepResult(
            observation=self.get_observations(),
            reward=self.judge(),
            done=self.is_done(),
            info={"last_action_success": self.last_action_success},
        )
        return step_result

    @classmethod
    def action_names(cls) -> Tuple[str, ...]:
        return cls._actions
