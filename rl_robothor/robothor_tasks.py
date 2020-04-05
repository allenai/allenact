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
from utils.system import LOGGER

# class RoboThorTask(Task[RoboThorEnvironment]):
#     def __init__(
#         self,
#         env: RoboThorEnvironment,
#         sensors: List[Sensor],
#         task_info: Dict[str, Any],
#         max_steps: int,
#         **kwargs
#     ) -> None:
#         super().__init__(
#             env=env, sensors=sensors, task_info=task_info, max_steps=max_steps, **kwargs
#         )
#
#         self._last_action: Optional[str] = None
#         self._last_action_ind: Optional[int] = None
#         self._last_action_success: Optional[bool] = None
#
#     @property
#     def last_action(self):
#         return self._last_action
#
#     @last_action.setter
#     def last_action(self, value: str):
#         self._last_action = value
#
#     @property
#     def last_action_success(self):
#         return self._last_action_success
#
#     @last_action_success.setter
#     def last_action_success(self, value: Optional[bool]):
#         self._last_action_success = value
#
#     # def _step(self, action: int) -> RLStepResult:
#     #     self._last_action_ind = action
#     #     self.last_action = self.action_names()[action]
#     #     self.last_action_success = None
#     #     step_result = super().step(action=action)
#     #     step_result.info["action"] = self._last_action_ind
#     #     step_result.info["action_success"] = self.last_action_success
#     #     return step_result
#
#     def render(self, mode: str = "rgb", *args, **kwargs) -> np.ndarray:
#         if mode == "rgb":
#             return self.env.current_frame["rgb"]
#         elif mode == "depth":
#             return self.env.current_frame["depth"]
#         else:
#             raise NotImplementedError()


class PointNavTask(Task[RoboThorEnvironment]):
    _actions = (MOVE_AHEAD, ROTATE_LEFT, ROTATE_RIGHT, END)

    def __init__(
        self,
        env: RoboThorEnvironment,
        sensors: List[Sensor],
        task_info: Dict[str, Any],
        max_steps: int,
        reward_configs: Dict[str, Any],
        **kwargs
    ) -> None:
        # print("task info in objectnavtask %s" % task_info)
        super().__init__(
            env=env, sensors=sensors, task_info=task_info, max_steps=max_steps, **kwargs
        )
        self.reward_configs = reward_configs
        self._took_end_action: bool = False
        self._success: Optional[bool] = False
        self._subsampled_locations_from_which_obj_visible = None
        self.episode_optimal_corners = self.env.path_corners(task_info["target"])  # assume it's valid (sampler must take care)!
        dist = self.env.path_corners_to_dist(self.episode_optimal_corners)
        if dist == float("inf"):
            dist = -1.0  # -1.0 for unreachable
            LOGGER.warning("No path for {} from {} to {}".format(self.env.scene_name, self.env.agent_state(), task_info["target"]))
        self.last_geodesic_distance = dist
        self._rewards = []
        self._distance_to_goal = []
        self._metrics = None
        # pose = self.env.agent_state()
        # self.path = [{k: pose[k] for k in ['x', 'y', 'z']}]
        self.path = []  # the initial coordinate will be directly taken from the optimal path

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
            pose = self.env.agent_state()
            self.path.append({k: pose[k] for k in ['x', 'y', 'z']})

        step_result = RLStepResult(
            observation=self.get_observations(),
            reward=self.judge(),
            done=self.is_done(),
            info={"last_action_success": self.last_action_success, "action": action},
        )
        return step_result

    def render(self, mode: str = "rgb", *args, **kwargs) -> np.ndarray:
        assert mode in ["rgb", "depth"], "only rgb and depth rendering is implemented"
        if mode == "rgb":
            return self.env.current_frame
        elif mode == "depth":
            return self.env.current_depth

    def _is_goal_in_range(self) -> Optional[bool]:
        tget = self.task_info["target"]
        # pose = self.env.agent_state()
        # dist = np.sqrt((pose['x'] - tget['x']) ** 2 + (pose['z'] - tget['z']) ** 2)
        dist = self.env.dist_to_point(tget)

        if -0.5 < dist <= 0.2:
            return True
        elif dist > 0.2:
            return False
        else:
            LOGGER.warning("No path for {} from {} to {}".format(
                self.env.scene_name,
                self.env.agent_state(),
                tget
            ))
            return None

    # def judge(self) -> float:
    #     reward = -0.01
    #
    #     geodesic_distance = self.env.dist_to_point(self.task_info['target'])
    #     if self.last_geodesic_distance > -0.5 and geodesic_distance > -0.5:  # (robothor limits)
    #         reward += self.last_geodesic_distance - geodesic_distance
    #     self.last_geodesic_distance = geodesic_distance
    #
    #     if self._took_end_action:
    #         reward += 10.0 if self._success else 0.0
    #
    #     self._rewards.append(float(reward))
    #
    #     return float(reward)

    def shaping(self) -> float:
        rew = 0.0

        if self.reward_configs["shaping_weight"] == 0.0:
            return rew

        geodesic_distance = self.env.dist_to_point(self.task_info['target'])
        if self.last_geodesic_distance > -0.5 and geodesic_distance > -0.5:  # (robothor limits)
            rew += self.last_geodesic_distance - geodesic_distance
            # if self.last_geodesic_distance > geodesic_distance:
            #     rew += self.reward_configs["delta_dist_reward_closer"]
            # elif self.last_geodesic_distance == geodesic_distance:
            #     rew += self.reward_configs["delta_dist_reward_same"]
            # else:
            #     rew += self.reward_configs["delta_dist_reward_further"]
        self.last_geodesic_distance = geodesic_distance

        # # ...and also exploring! We won't be able to hit the optimal path in test
        # old_visited = len(self.visited)
        # self.visited.add(
        #     self.env.agent_to_grid(xz_subsampling=4, rot_subsampling=3)
        # )  # squares of 1 m2, sectors of 90 deg
        # rew += self.reward_configs["exploration_shaping_weight"] * (
        #     len(self.visited) - old_visited
        # )

        return rew * self.reward_configs["shaping_weight"]

    def judge(self) -> float:
        """ Judge the last event. """
        reward = self.reward_configs["step_penalty"]

        reward += self.shaping()

        # if not self.last_action_success:
        #     reward += self.reward_configs["unsuccessful_action_penalty"]

        if self._took_end_action:
            if self._success is not None:
                reward += (
                    self.reward_configs["goal_success_reward"]
                    if self._success
                    else self.reward_configs["failed_stop_reward"]
                )

        self._rewards.append(float(reward))

        return float(reward)

    def spl(self):
        if len(self.episode_optimal_corners) <= 1 or self._success is None:
            return None

        if not self._success:
            return 0.0

        # pose = self.env.agent_state()
        res = compute_single_spl(self.path, self.episode_optimal_corners, self._success)  # no controller involved

        # # self.env.step({"action": "TeleportFull", **pose})
        # new_pose = self.env.agent_state()
        # assert abs(new_pose['x'] - pose['x']) < 1e-5, "wrong x"
        # assert abs(new_pose['y'] - pose['y']) < 1e-5, "wrong y"
        # assert abs(new_pose['z'] - pose['z']) < 1e-5, "wrong z"
        # assert abs(new_pose['rotation']['y'] - pose['rotation']['y']) < 1e-5, "wrong rotation y"
        # assert abs(new_pose['horizon'] - pose['horizon']) < 1e-5, "wrong horizon"

        return res

    def dist_to_target(self):
        res = self.env.dist_to_point(self.task_info['target'])
        return res if res > -0.5 else None

    def metrics(self) -> Dict[str, Any]:
        if not self.is_done():
            return {}
        else:
            total_reward = float(np.sum(self._rewards))
            self._rewards = []

            if self._success is None:
                return {}

            dist2tget = self.dist_to_target()
            if dist2tget is None:
                return {}

            spl = self.spl()
            if spl is None:
                return {}

            return {
                "success": self._success,  # False also if no path to target
                "ep_length": self.num_steps_taken(),
                "total_reward": total_reward,
                "dist_to_target": dist2tget,
                "spl": spl,
            }


class ObjectNavTask(Task[RoboThorEnvironment]):
    _actions = (MOVE_AHEAD, ROTATE_LEFT, ROTATE_RIGHT, END, LOOK_UP, LOOK_DOWN)

    def __init__(
        self,
        env: RoboThorEnvironment,
        sensors: List[Sensor],
        task_info: Dict[str, Any],
        max_steps: int,
        reward_configs: Dict[str, Any],
        **kwargs
    ) -> None:
        # print("task info in objectnavtask %s" % task_info)
        super().__init__(
            env=env, sensors=sensors, task_info=task_info, max_steps=max_steps, **kwargs
        )
        self.reward_configs = reward_configs
        self._took_end_action: bool = False
        self._success: Optional[bool] = False
        self._subsampled_locations_from_which_obj_visible = None
        self.episode_optimal_corners = self.env.path_corners(task_info["object_type"])  # assume it's valid (sampler must take care)!
        self.mirror = task_info['mirrored']
        dist = self.env.path_corners_to_dist(self.episode_optimal_corners)
        if dist == float("inf"):
            dist = -1.0  # -1.0 for unreachable
        self.last_geodesic_distance = dist
        self._rewards = []
        self._distance_to_goal = []
        self._metrics = None
        # pose = self.env.agent_state()
        # self.path = [{k: pose[k] for k in ['x', 'y', 'z']}]
        self.path = []  # the initial coordinate will be directly taken from the optimal path

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

        if self.mirror:
            if action_str == ROTATE_RIGHT:
                action_str = ROTATE_LEFT
            elif action_str == ROTATE_LEFT:
                action_str = ROTATE_RIGHT

        if action_str == END:
            self._took_end_action = True
            self._success = self._is_goal_in_range()
            self.last_action_success = self._success
        else:
            self.env.step({"action": action_str})
            self.last_action_success = self.env.last_action_success
            pose = self.env.agent_state()
            self.path.append({k: pose[k] for k in ['x', 'y', 'z']})

        step_result = RLStepResult(
            observation=self.get_observations(),
            reward=self.judge(),
            done=self.is_done(),
            info={"last_action_success": self.last_action_success, "action": action},
        )
        return step_result

    def render(self, mode: str = "rgb", *args, **kwargs) -> np.ndarray:
        assert mode in ["rgb", "depth"], "only rgb and depth rendering is implemented"
        if mode == "rgb":
            frame = self.env.current_frame.copy()
        elif mode == "depth":
            frame = self.env.current_depth.copy()
        if self.mirror:
            frame = frame[:, ::-1, :].copy()  # horizontal flip
            # print("mirrored render")
        return frame

    def _is_goal_in_range(self) -> bool:
        return any(
            o["objectType"] == self.task_info["object_type"]
            for o in self.env.visible_objects()
        )

    # def judge(self) -> float:
    #     reward = -0.01
    #
    #     geodesic_distance = self.env.dist_to_object(self.task_info["object_type"])
    #     if self.last_geodesic_distance > -0.5 and geodesic_distance > -0.5:  # (robothor limits)
    #         reward += self.last_geodesic_distance - geodesic_distance
    #     self.last_geodesic_distance = geodesic_distance
    #
    #     if self._took_end_action:
    #         reward += 10.0 if self._success else 0.0
    #
    #     self._rewards.append(float(reward))
    #
    #     return float(reward)

    def shaping(self) -> float:
        rew = 0.0

        if self.reward_configs["shaping_weight"] == 0.0:
            return rew

        # geodesic_distance = self.env.dist_to_object(self.task_info["object_type"])
        # if self.last_geodesic_distance > -0.5 and geodesic_distance > -0.5:  # (robothor limits)
        #     if self.last_geodesic_distance > geodesic_distance:
        #         rew += self.reward_configs["delta_dist_reward_closer"]
        #     elif self.last_geodesic_distance == geodesic_distance:
        #         rew += self.reward_configs["delta_dist_reward_same"]
        #     else:
        #         rew += self.reward_configs["delta_dist_reward_further"]
        # self.last_geodesic_distance = geodesic_distance

        geodesic_distance = self.env.dist_to_object(self.task_info["object_type"])
        if self.last_geodesic_distance > -0.5 and geodesic_distance > -0.5:  # (robothor limits)
            rew += self.last_geodesic_distance - geodesic_distance
        self.last_geodesic_distance = geodesic_distance

        # # ...and also exploring! We won't be able to hit the optimal path in test
        # old_visited = len(self.visited)
        # self.visited.add(
        #     self.env.agent_to_grid(xz_subsampling=4, rot_subsampling=3)
        # )  # squares of 1 m2, sectors of 90 deg
        # rew += self.reward_configs["exploration_shaping_weight"] * (
        #     len(self.visited) - old_visited
        # )

        return rew * self.reward_configs["shaping_weight"]

    def judge(self) -> float:
        """ Judge the last event. """
        reward = self.reward_configs["step_penalty"]

        reward += self.shaping()

        # if not self.last_action_success:
        #     reward += self.reward_configs["unsuccessful_action_penalty"]

        if self._took_end_action:
            reward += (
                self.reward_configs["goal_success_reward"]
                if self._success
                else self.reward_configs["failed_stop_reward"]
            )

        self._rewards.append(float(reward))

        return float(reward)

    def spl(self):
        pose = self.env.agent_state()
        res = compute_single_spl(self.path, self.episode_optimal_corners, self._success)
        # self.env.step({"action": "TeleportFull", **pose})
        return res

    def get_observations(self) -> Any:
        obs = self.sensor_suite.get_observations(env=self.env, task=self)
        if self.mirror:
            # flipped = []
            for o in obs:
                if ('rgb' in o or 'depth' in o) and isinstance(obs[o], np.ndarray) and len(obs[o].shape) == 3:  # heuristic to determine this is a visual sensor
                    obs[o] = obs[o][:, ::-1, :].copy()  # horizontal flip
                    # flipped.append(o)
            # print('flipped {}'.format(flipped))
        return obs

    def metrics(self) -> Dict[str, Any]:
        if not self.is_done():
            return {}
        else:
            metrics = {
                "success": self._success,
                "ep_length": self.num_steps_taken(),
                "total_reward": np.sum(self._rewards),
                "spl": self.spl() if len(self.episode_optimal_corners) > 1 else 0.0,
                "task_info": self.task_info,
            }
            self._rewards = []
            return metrics
