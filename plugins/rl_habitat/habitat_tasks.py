# TODO: @klemenkotar please fix all type errors

from typing import Tuple, List, Dict, Any, Optional

import gym
import numpy as np
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower

from rl_base.common import RLStepResult
from rl_base.sensor import Sensor
from rl_base.task import Task
from plugins.rl_habitat.habitat_constants import (
    MOVE_AHEAD,
    ROTATE_LEFT,
    ROTATE_RIGHT,
    END,
    LOOK_UP,
    LOOK_DOWN,
)
from plugins.rl_habitat.habitat_environment import HabitatEnvironment


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
        self._actions_taken = []
        self._positions = []
        pos = self.get_observations()["agent_position_and_rotation"]
        self._positions.append(
            {"x": pos[0], "y": pos[1], "z": pos[2], "rotation": pos[3]}
        )
        ep = self.env.get_current_episode()
        # Extract the scene name from the scene path and append the episode id to generate
        # a globally unique episode_id
        self._episode_id = ep.scene_id[-15:-4] + "_" + ep.episode_id

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

    def render(self, mode: str = "rgb", *args, **kwargs) -> np.ndarray:
        if mode == "rgb":
            return self.env.current_frame["rgb"]
        elif mode == "depth":
            return self.env.current_frame["depth"]
        else:
            raise NotImplementedError()


class PointNavTask(Task[HabitatEnvironment]):
    _actions = (MOVE_AHEAD, ROTATE_LEFT, ROTATE_RIGHT, END)

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
        self._took_end_action: bool = False
        self._success: Optional[bool] = False
        self._subsampled_locations_from_which_obj_visible = None

        # Get the geodesic distance to target from the environemnt and make sure it is
        # a valid value
        self.last_geodesic_distance = self.env.env.get_metrics()["distance_to_goal"]
        if (
            self.last_geodesic_distance is None
            or self.last_geodesic_distance in [float("-inf"), float("inf")]
            or np.isnan(self.last_geodesic_distance)
        ):
            self.last_geodesic_distance = 0.0

        self._shortest_path_follower = ShortestPathFollower(
            env.env.sim, env.env._config.TASK.SUCCESS_DISTANCE, False
        )
        self._shortest_path_follower.mode = "geodesic_path"

        self._rewards: List[float] = []
        self._metrics = None

    @property
    def action_space(self):
        return gym.spaces.Discrete(len(self._actions))

    def reached_terminal_state(self) -> bool:
        return self.env.env.episode_over

    @classmethod
    def class_action_names(cls) -> Tuple[str, ...]:
        return cls._actions

    def close(self) -> None:
        self.env.stop()

    def _step(self, action: int) -> RLStepResult:
        action_str = self.class_action_names()[action]

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
        return self.env.current_frame["rgb"]

    def _is_goal_in_range(self) -> bool:
        # The habitat simulator will return an SPL value of 0.0 whenever the goal is not in range
        return bool(self.env.env.get_metrics()["spl"])

    def judge(self) -> float:
        reward = -0.01

        new_geodesic_distance = self.env.env.get_metrics()["distance_to_goal"]
        if (
            new_geodesic_distance is None
            or new_geodesic_distance in [float("-inf"), float("inf")]
            or np.isnan(new_geodesic_distance)
        ):
            new_geodesic_distance = self.last_geodesic_distance
        delta_distance_reward = self.last_geodesic_distance - new_geodesic_distance
        reward += delta_distance_reward
        self.last_geodesic_distance = new_geodesic_distance

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
                "spl": _metrics["spl"] if _metrics["spl"] is not None else 0.0,
            }
            self._rewards = []
            return metrics

    def query_expert(self, **kwargs) -> Tuple[int, bool]:
        if self._is_goal_in_range():
            return self.class_action_names().index(END), True

        target = self.task_info["target"]
        action = self._shortest_path_follower.get_next_action(target)
        return action, action is not None


class ObjectNavTask(HabitatTask):
    _actions = (MOVE_AHEAD, ROTATE_LEFT, ROTATE_RIGHT, END, LOOK_UP, LOOK_DOWN)

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
        self._took_end_action: bool = False
        self._success: Optional[bool] = False
        self._subsampled_locations_from_which_obj_visible = None

        # Get the geodesic distance to target from the environemnt and make sure it is
        # a valid value
        self.last_geodesic_distance = self.env.env.get_metrics()["distance_to_goal"]
        if (
            self.last_geodesic_distance is None
            or self.last_geodesic_distance in [float("-inf"), float("inf")]
            or np.isnan(self.last_geodesic_distance)
        ):
            self.last_geodesic_distance = 0.0
        self._min_distance_to_goal = self.last_geodesic_distance
        self._num_invalid_actions = 0

        self._shortest_path_follower = ShortestPathFollower(
            env.env.sim, env.env._config.TASK.SUCCESS_DISTANCE, False
        )
        self._shortest_path_follower.mode = "geodesic_path"

        self._rewards: List[float] = []
        self._metrics = None
        self.task_info["episode_id"] = self._episode_id
        self.task_info["target_position"] = {
            "x": self.task_info["target"][0],
            "y": self.task_info["target"][1],
            "z": self.task_info["target"][2],
        }

        self._coverage_map = np.zeros((150, 150))

    @property
    def action_space(self):
        return gym.spaces.Discrete(len(self._actions))

    def reached_terminal_state(self) -> bool:
        return self.env.env.episode_over

    @classmethod
    def class_action_names(cls) -> Tuple[str, ...]:
        return cls._actions

    def close(self) -> None:
        self.env.stop()

    def _step(self, action: int) -> RLStepResult:
        old_pos = self.get_observations()["agent_position_and_rotation"]

        action_str = self.action_names()[action]
        self._actions_taken.append(action_str)

        self.env.step({"action": action_str})

        # if action_str != END:
        #     self.env.step({"action": action_str})

        # if self.env.env.get_metrics()['distance_to_goal'] <= 0.2:
        #     self._took_end_action = True
        #     self._success = self.env.env.get_metrics()['distance_to_goal'] <= 0.2
        #     self.last_action_success = self._success
        # else:
        #     self.last_action_success = self.env.last_action_success

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
        new_pos = self.get_observations()["agent_position_and_rotation"]
        if np.all(old_pos == new_pos):
            self._num_invalid_actions += 1

        pos = self.get_observations()["agent_position_and_rotation"]
        self._positions.append(
            {"x": pos[0], "y": pos[1], "z": pos[2], "rotation": pos[3]}
        )

        return step_result

    def render(self, mode: str = "rgb", *args, **kwargs) -> np.ndarray:
        assert mode in ["rgb", "depth"], "only rgb and depth rendering is implemented"
        return self.env.current_frame["rgb"]

    def _is_goal_in_range(self) -> bool:
        # The habitat simulator will return an SPL value of 0.0 whenever the goal is not in range
        return bool(self.env.env.get_metrics()["spl"])

    def judge(self) -> float:
        # Set default reward
        reward = -0.01

        # Get geodesic distance reward
        new_geodesic_distance = self.env.env.get_metrics()["distance_to_goal"]
        self._min_distance_to_goal = min(
            new_geodesic_distance, self._min_distance_to_goal
        )
        if (
            new_geodesic_distance is None
            or new_geodesic_distance in [float("-inf"), float("inf")]
            or np.isnan(new_geodesic_distance)
        ):
            new_geodesic_distance = self.last_geodesic_distance
        delta_distance_reward = self.last_geodesic_distance - new_geodesic_distance
        reward += delta_distance_reward

        if self._took_end_action:
            reward += 10.0 if self._success else 0.0

        # Get success reward
        self._rewards.append(float(reward))
        self.last_geodesic_distance = new_geodesic_distance

        # # Get coverage reward
        # pos = self.get_observations()["agent_position_and_rotation"]
        # # align current position with center of map
        # x = int(pos[0] + 75)
        # y = int(pos[2] + 75)
        # if self._coverage_map[x, y] == 0:
        #     self._coverage_map[x, y] = 1
        #     reward += 0.1
        # else:
        #     reward -= 0.0

        return float(reward)

    def metrics(self) -> Dict[str, Any]:
        self.task_info["taken_actions"] = self._actions_taken
        self.task_info["action_names"] = self.action_names()
        self.task_info["followed_path"] = self._positions
        if not self.is_done():
            return {}
        else:
            _metrics = self.env.env.get_metrics()
            metrics = {
                "success": self._success,
                "ep_length": self.num_steps_taken(),
                "total_reward": np.sum(self._rewards),
                "spl": _metrics["spl"] if _metrics["spl"] is not None else 0.0,
                "min_distance_to_target": self._min_distance_to_goal,
                "num_invalid_actions": self._num_invalid_actions,
                "task_info": self.task_info,
            }
            self._rewards = []
            return metrics

    def query_expert(self, **kwargs) -> Tuple[int, bool]:
        if self._is_goal_in_range():
            return self.class_action_names().index(END), True

        target = self.task_info["target"]
        action = self._shortest_path_follower.get_next_action(target)
        return action, action is not None
