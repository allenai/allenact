from abc import ABC
from typing import Tuple, List, Dict, Any, Optional, Union, Sequence, cast

import gym
import numpy as np
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.sims.habitat_simulator.habitat_simulator import HabitatSim
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower

from allenact.base_abstractions.misc import RLStepResult
from allenact.base_abstractions.sensor import Sensor
from allenact.base_abstractions.task import Task
from allenact.utils.system import get_logger
from allenact_plugins.habitat_plugin.habitat_constants import (
    MOVE_AHEAD,
    ROTATE_LEFT,
    ROTATE_RIGHT,
    END,
    LOOK_UP,
    LOOK_DOWN,
)
from allenact_plugins.habitat_plugin.habitat_environment import HabitatEnvironment
from allenact_plugins.habitat_plugin.habitat_sensors import (
    AgentCoordinatesSensorHabitat,
)


class HabitatTask(Task[HabitatEnvironment], ABC):
    def __init__(
        self,
        env: HabitatEnvironment,
        sensors: List[Sensor],
        task_info: Dict[str, Any],
        max_steps: int,
        **kwargs,
    ) -> None:
        super().__init__(
            env=env, sensors=sensors, task_info=task_info, max_steps=max_steps, **kwargs
        )

        self._last_action: Optional[str] = None
        self._last_action_ind: Optional[int] = None
        self._last_action_success: Optional[bool] = None
        self._actions_taken: List[str] = []
        self._positions = []
        pos = self.get_agent_position_and_rotation()
        self._positions.append(
            {"x": pos[0], "y": pos[1], "z": pos[2], "rotation": pos[3]}
        )
        ep = self.env.get_current_episode()
        # Extract the scene name from the scene path and append the episode id to generate
        # a globally unique episode_id
        self._episode_id = ep.scene_id.split("/")[-1][:-4] + "_" + ep.episode_id

    def get_agent_position_and_rotation(self):
        return AgentCoordinatesSensorHabitat.get_observation(self.env, self)

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
        failed_end_reward: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__(
            env=env, sensors=sensors, task_info=task_info, max_steps=max_steps, **kwargs
        )
        self._took_end_action: bool = False
        self._success: Optional[bool] = False
        self._subsampled_locations_from_which_obj_visible = None

        # Get the geodesic distance to target from the environment and make sure it is
        # a valid value
        self.last_geodesic_distance = self.current_geodesic_dist_to_target()
        self.start_distance = self.last_geodesic_distance
        assert self.last_geodesic_distance is not None

        # noinspection PyProtectedMember
        self._shortest_path_follower = ShortestPathFollower(
            cast(HabitatSim, env.env.sim), env.env._config.TASK.SUCCESS_DISTANCE, False
        )
        self._shortest_path_follower.mode = "geodesic_path"

        self._rewards: List[float] = []
        self._metrics = None
        self.failed_end_reward = failed_end_reward

    def current_geodesic_dist_to_target(self) -> Optional[float]:
        metrics = self.env.env.get_metrics()
        if metrics["distance_to_goal"] is None:
            habitat_env = self.env.env
            habitat_env.task.measurements.update_measures(
                episode=habitat_env.current_episode, action=None, task=habitat_env.task
            )
            metrics = self.env.env.get_metrics()

        return metrics["distance_to_goal"]

    @property
    def action_space(self):
        return gym.spaces.Discrete(len(self._actions))

    def reached_terminal_state(self) -> bool:
        return self.env.env.episode_over

    @classmethod
    def class_action_names(cls, **kwargs) -> Tuple[str, ...]:
        return cls._actions

    def close(self) -> None:
        self.env.stop()

    def _step(self, action: Union[int, Sequence[int]]) -> RLStepResult:
        assert isinstance(action, int)
        action = cast(int, action)

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
        return (
            self.current_geodesic_dist_to_target() <= self.task_info["distance_to_goal"]
        )

    def judge(self) -> float:
        reward = -0.01

        new_geodesic_distance = self.current_geodesic_dist_to_target()
        if self.last_geodesic_distance is None:
            self.last_geodesic_distance = new_geodesic_distance

        if self.last_geodesic_distance is not None:
            if (
                new_geodesic_distance is None
                or new_geodesic_distance in [float("-inf"), float("inf")]
                or np.isnan(new_geodesic_distance)
            ):
                new_geodesic_distance = self.last_geodesic_distance
            delta_distance_reward = self.last_geodesic_distance - new_geodesic_distance
            reward += delta_distance_reward
            self.last_geodesic_distance = new_geodesic_distance

            if self.is_done():
                reward += 10.0 if self._success else self.failed_end_reward
        else:
            get_logger().warning("Could not get geodesic distance from habitat env.")

        self._rewards.append(float(reward))

        return float(reward)

    def metrics(self) -> Dict[str, Any]:
        if not self.is_done():
            return {}

        _metrics = self.env.env.get_metrics()
        metrics = {
            **super(PointNavTask, self).metrics(),
            "success": 1 * self._success,
            "ep_length": self.num_steps_taken(),
            "reward": np.sum(self._rewards),
            "spl": _metrics["spl"] if _metrics["spl"] is not None else 0.0,
            "dist_to_target": self.current_geodesic_dist_to_target(),
        }
        self._rewards = []
        return metrics

    def query_expert(self, **kwargs) -> Tuple[int, bool]:
        if self._is_goal_in_range():
            return self.class_action_names().index(END), True

        target = self.task_info["target"]
        habitat_action = self._shortest_path_follower.get_next_action(target)
        if habitat_action == HabitatSimActions.MOVE_FORWARD:
            return self.class_action_names().index(MOVE_AHEAD), True
        elif habitat_action == HabitatSimActions.TURN_LEFT:
            return self.class_action_names().index(ROTATE_LEFT), True
        elif habitat_action == HabitatSimActions.TURN_RIGHT:
            return self.class_action_names().index(ROTATE_RIGHT), True
        else:
            return 0, False


class ObjectNavTask(HabitatTask):
    _actions = (MOVE_AHEAD, ROTATE_LEFT, ROTATE_RIGHT, END, LOOK_UP, LOOK_DOWN)

    def __init__(
        self,
        env: HabitatEnvironment,
        sensors: List[Sensor],
        task_info: Dict[str, Any],
        max_steps: int,
        look_constraints: Optional[Tuple[int, int]] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            env=env, sensors=sensors, task_info=task_info, max_steps=max_steps, **kwargs
        )
        self.look_constraints = look_constraints
        self._look_state = 0

        self._took_end_action: bool = False
        self._success: Optional[bool] = False
        self._subsampled_locations_from_which_obj_visible = None

        # Get the geodesic distance to target from the environemnt and make sure it is
        # a valid value
        self.last_geodesic_distance = self.current_geodesic_dist_to_target()
        assert not (
            self.last_geodesic_distance is None
            or self.last_geodesic_distance in [float("-inf"), float("inf")]
            or np.isnan(self.last_geodesic_distance)
        ), "Bad geodesic distance"
        self._min_distance_to_goal = self.last_geodesic_distance
        self._num_invalid_actions = 0

        # noinspection PyProtectedMember
        self._shortest_path_follower = ShortestPathFollower(
            env.env.sim, env.env._config.TASK.SUCCESS.SUCCESS_DISTANCE, False
        )
        self._shortest_path_follower.mode = "geodesic_path"

        self._rewards: List[float] = []
        self._metrics = None
        self.task_info["episode_id"] = self._episode_id

    @property
    def action_space(self):
        return gym.spaces.Discrete(len(self._actions))

    def reached_terminal_state(self) -> bool:
        return self.env.env.episode_over

    @classmethod
    def class_action_names(cls, **kwargs) -> Tuple[str, ...]:
        return cls._actions

    def action_names(self, **kwargs) -> Tuple[str, ...]:
        return self._actions

    def close(self) -> None:
        self.env.stop()

    def current_geodesic_dist_to_target(self) -> Optional[float]:
        metrics = self.env.env.get_metrics()
        if metrics["distance_to_goal"] is None:
            habitat_env = self.env.env
            habitat_env.task.measurements.update_measures(
                episode=habitat_env.current_episode, action=None, task=habitat_env.task
            )
            metrics = self.env.env.get_metrics()

        return metrics["distance_to_goal"]

    def _step(self, action: Union[int, Sequence[int]]) -> RLStepResult:
        assert isinstance(action, int)
        action = cast(int, action)

        old_pos = self.get_agent_position_and_rotation()

        action_str = self.action_names()[action]
        self._actions_taken.append(action_str)

        skip_action = False
        if self.look_constraints is not None:
            max_look_up, max_look_down = self.look_constraints

            if action_str == LOOK_UP:
                num_look_ups = self._look_state
                # assert num_look_ups <= max_look_up
                skip_action = num_look_ups >= max_look_up
                self._look_state += 1

            if action_str == LOOK_DOWN:
                num_look_downs = -self._look_state
                # assert num_look_downs <= max_look_down
                skip_action = num_look_downs >= max_look_down
                self._look_state -= 1

            self._look_state = min(max(self._look_state, -max_look_down), max_look_up)

        if not skip_action:
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
        new_pos = self.get_agent_position_and_rotation()
        if np.all(old_pos == new_pos):
            self._num_invalid_actions += 1

        pos = self.get_agent_position_and_rotation()
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
        new_geodesic_distance = self.current_geodesic_dist_to_target()
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
