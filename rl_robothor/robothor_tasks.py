from typing import Tuple, List, Dict, Any, Optional

import gym
import numpy as np
from ai2thor.util.metrics import compute_single_spl

from rl_robothor.robothor_constants import (
    MOVE_AHEAD,
    ROTATE_LEFT,
    ROTATE_RIGHT,
    END,
    LOOK_UP,
    LOOK_DOWN,
)
from rl_robothor.robothor_environment import RoboThorEnvironment
from rl_base.common import RLStepResult
from rl_base.sensor import Sensor
from rl_base.task import Task


class RoboThorTask(Task[RoboThorEnvironment]):
    def __init__(
        self,
        env: RoboThorEnvironment,
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
        step_result = super().step(action=action)
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


class PointNavTask(Task[RoboThorEnvironment]):
    _actions = (MOVE_AHEAD, ROTATE_LEFT, ROTATE_RIGHT, END)

    def __init__(
        self,
        env: RoboThorEnvironment,
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
        self.episode_optimal_corners = self.env.path_corners(task_info["target"])  # assume it's valid (sampler must take care)!
        self.last_geodesic_distance = self.env.path_corners_to_dist(self.episode_optimal_corners)
        self._rewards = []
        self._distance_to_goal = []
        self._metrics = None
        pose = self.env.agent_state()
        self.path = [{k: pose[k] for k in ['x', 'y', 'z']}]

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

        self.env.step({"action": action_str})

        pose = self.env.agent_state()
        self.path.append({k: pose[k] for k in ['x', 'y', 'z']})

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
        tget = self.task_info["target"]
        # pose = self.env.agent_state()
        # dist = np.sqrt((pose['x'] - tget['x']) ** 2 + (pose['z'] - tget['z']) ** 2)
        dist = self.env.dist_to_point(**tget)
        return -0.5 < dist <= 0.2

    def judge(self) -> float:
        reward = -0.01

        geodesic_distance = self.env.dist_to_point(self.task_info['target'])
        delta_distance_reward = self.last_geodesic_distance - geodesic_distance
        reward += delta_distance_reward
        self.last_geodesic_distance = geodesic_distance

        if self._took_end_action:
            reward += 10.0 if self._success else 0.0

        self._rewards.append(float(reward))

        return float(reward)

    def spl(self):
        pose = self.env.agent_state()
        res = compute_single_spl(self.path, self.episode_optimal_corners, self._success)
        self.env.step({"action": "TeleportFull", **pose})
        return res

    def metrics(self) -> Dict[str, Any]:
        if not self.is_done():
            return {}
        else:
            metrics = {
                "success": self._success,
                "ep_length": self.num_steps_taken(),
                "total_reward": np.sum(self._rewards),
                "spl": self.spl() if len(self.episode_optimal_corners) > 0 else 0.0
            }
            self._rewards = []
            return metrics


class ObjectNavTask(Task[RoboThorEnvironment]):
    _actions = (MOVE_AHEAD, ROTATE_LEFT, ROTATE_RIGHT, END, LOOK_UP, LOOK_DOWN)

    def __init__(
        self,
        env: RoboThorEnvironment,
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
        self.episode_optimal_corners = self.env.path_corners(task_info["object_type"])  # assume it's valid (sampler must take care)!
        self.last_geodesic_distance = self.env.path_corners_to_dist(self.episode_optimal_corners)
        self._rewards = []
        self._distance_to_goal = []
        self._metrics = None
        pose = self.env.agent_state()
        self.path = [{k: pose[k] for k in ['x', 'y', 'z']}]

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

        self.env.step({"action": action_str})

        pose = self.env.agent_state()
        self.path.append({k: pose[k] for k in ['x', 'y', 'z']})

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
        if mode == "rgb":
            return self.env.current_frame
        elif mode == "depth":
            return self.env.current_depth

    def _is_goal_in_range(self) -> bool:
        return any(
            o["objectType"] == self.task_info["object_type"]
            for o in self.env.visible_objects()
        )

    def judge(self) -> float:
        reward = -0.01

        geodesic_distance = self.env.dist_to_object(self.task_info["object_type"])
        if self.last_geodesic_distance > -0.5 and geodesic_distance > -0.5:  # (robothor limits)
            reward += self.last_geodesic_distance - geodesic_distance
        self.last_geodesic_distance = geodesic_distance

        if self._took_end_action:
            reward += 10.0 if self._success else 0.0

        self._rewards.append(float(reward))

        return float(reward)

    def spl(self):
        pose = self.env.agent_state()
        res = compute_single_spl(self.path, self.episode_optimal_corners, self._success)
        self.env.step({"action": "TeleportFull", **pose})
        return res

    def metrics(self) -> Dict[str, Any]:
        if not self.is_done():
            return {}
        else:
            metrics = {
                "success": self._success,
                "ep_length": self.num_steps_taken(),
                "total_reward": np.sum(self._rewards),
                "spl": self.spl() if len(self.episode_optimal_corners) > 0 else 0.0
            }
            self._rewards = []
            return metrics
