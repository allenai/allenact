import random
import warnings
from typing import Dict, Tuple, List, Any, Optional
import os
import pickle

import gym
import numpy as np
import networkx as nx

from rl_ai2thor.ai2thor_constants import (
    MOVE_AHEAD,
    ROTATE_LEFT,
    ROTATE_RIGHT,
    LOOK_DOWN,
    LOOK_UP,
    END,
)
from rl_ai2thor.ai2thor_environment import AI2ThorEnvironment
from rl_ai2thor.ai2thor_util import round_to_factor
from rl_base.common import RLStepResult
from rl_base.sensor import Sensor
from rl_base.task import Task


class ObjectNavTask(Task[AI2ThorEnvironment]):
    """Defines the object navigation task in AI2-THOR.

    In object navigation an agent is randomly initialized into an AI2-THOR scene and must
    find an object of a given type (e.g. tomato, television, etc). An object is considered
    found if the agent takes an `End` action and the object is visible to the agent (see
    [here](https://ai2thor.allenai.org/documentation/concepts) for a definition of visibiliy
    in AI2-THOR).

    The actions available to an agent in this task are:

    1. Move ahead
        * Moves agent ahead by 0.25 meters.
    1. Rotate left / rotate right
        * Rotates the agent by 90 degrees counter-clockwise / clockwise.
    1. Look down / look up
        * Changes agent view angle by 30 degrees up or down. An agent cannot look more than 30
          degrees above horizontal or less than 60 degrees below horizontal.
    1. End
        * Ends the task and the agent receives a positive reward if the object type is visible to the agent,
        otherwise it receives a negative reward.

    # Attributes

    env : The ai2thor environment.
    sensor_suite: Collection of sensors formed from the `sensors` argument in the initializer.
    task_info : The task info. Must contain a field "object_type" that specifies, as a string,
        the goal object type.
    max_steps : The maximum number of steps an agent can take an in the task before it is considered failed.
    observation_space: The observation space returned on each step from the sensors.
    """

    _actions = (MOVE_AHEAD, ROTATE_LEFT, ROTATE_RIGHT, LOOK_DOWN, LOOK_UP, END)

    _CACHED_LOCATIONS_FROM_WHICH_OBJECT_IS_VISIBLE: Dict[
        Tuple[str, str], List[Tuple[float, float, int, int]]
    ] = {}

    def __init__(
        self,
        env: AI2ThorEnvironment,
        sensors: List[Sensor],
        task_info: Dict[str, Any],
        max_steps: int,
        **kwargs
    ) -> None:
        """Initializer.

        See class documentation for parameter definitions.
        """
        super().__init__(
            env=env, sensors=sensors, task_info=task_info, max_steps=max_steps, **kwargs
        )
        self._took_end_action: bool = False
        self._success: Optional[bool] = False
        self._subsampled_locations_from_which_obj_visible: Optional[
            List[Tuple[float, float, int, int]]
        ] = None

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
            self._success = self._is_goal_object_visible()
            self.last_action_success = self._success
        else:
            self.env.step({"action": action_str})
            self.last_action_success = self.env.last_action_success

            if (
                not self.last_action_success
            ) and self._CACHED_LOCATIONS_FROM_WHICH_OBJECT_IS_VISIBLE is not None:
                self.env.update_graph_with_failed_action(failed_action=action_str)

        step_result = RLStepResult(
            observation=self.get_observations(),
            reward=self.judge(),
            done=self.is_done(),
            info={"last_action_success": self.last_action_success},
        )
        return step_result

    def render(self, mode: str = "rgb", *args, **kwargs) -> np.ndarray:
        assert mode == "rgb", "only rgb rendering is implemented"
        return self.env.current_frame

    def _is_goal_object_visible(self) -> bool:
        """Is the goal object currently visible?"""
        return any(
            o["objectType"] == self.task_info["object_type"]
            for o in self.env.visible_objects()
        )

    def judge(self) -> float:
        """Compute the reward after having taken a step."""
        reward = -0.01

        if not self.last_action_success:
            reward += -0.03

        if self._took_end_action:
            reward += 1.0 if self._success else -1.0

        return float(reward)

    def metrics(self) -> Dict[str, Any]:
        if not self.is_done():
            return {}
        else:
            return {"success": self._success, **super(ObjectNavTask, self).metrics()}

    def query_expert(self) -> Tuple[int, bool]:
        target = self.task_info["object_type"]

        if self._is_goal_object_visible():
            return self.action_names().index(END), True
        else:
            key = (self.env.scene_name, target)
            if self._subsampled_locations_from_which_obj_visible is None:
                if key not in self._CACHED_LOCATIONS_FROM_WHICH_OBJECT_IS_VISIBLE:
                    obj_ids: List[str] = []
                    obj_ids.extend(
                        o["objectId"]
                        for o in self.env.last_event.metadata["objects"]
                        if o["objectType"] == target
                    )

                    assert len(obj_ids) != 0, "No objects to get an expert path to."

                    locations_from_which_object_is_visible: List[
                        Tuple[float, float, int, int]
                    ] = []
                    y = self.env.last_event.metadata["agent"]["position"]["y"]
                    positions_to_check_interactionable_from = [
                        {"x": x, "y": y, "z": z}
                        for x, z in set((x, z) for x, z, _, _ in self.env.graph.nodes)
                    ]
                    for obj_id in set(obj_ids):
                        self.env.controller.step(
                            {
                                "action": "PositionsFromWhichItemIsInteractable",
                                "objectId": obj_id,
                                "positions": positions_to_check_interactionable_from,
                            }
                        )
                        assert (
                            self.env.last_action_success
                        ), "Could not get positions from which item was interactable."

                        returned = self.env.last_event.metadata["actionReturn"]
                        locations_from_which_object_is_visible.extend(
                            (
                                round(x, 2),
                                round(z, 2),
                                round_to_factor(rot, 90) % 360,
                                round_to_factor(hor, 30) % 360,
                            )
                            for x, z, rot, hor, standing in zip(
                                returned["x"],
                                returned["z"],
                                returned["rotation"],
                                returned["horizon"],
                                returned["standing"],
                            )
                            if standing == 1
                        )

                    self._CACHED_LOCATIONS_FROM_WHICH_OBJECT_IS_VISIBLE[
                        key
                    ] = locations_from_which_object_is_visible

                self._subsampled_locations_from_which_obj_visible = self._CACHED_LOCATIONS_FROM_WHICH_OBJECT_IS_VISIBLE[
                    key
                ]
                if len(self._subsampled_locations_from_which_obj_visible) > 5:
                    self._subsampled_locations_from_which_obj_visible = random.sample(
                        self._CACHED_LOCATIONS_FROM_WHICH_OBJECT_IS_VISIBLE[key], 5
                    )

            current_loc_key = self.env.get_key(self.env.last_event.metadata["agent"])
            paths = []

            for goal_key in self._subsampled_locations_from_which_obj_visible:
                path = self.env.shortest_state_path(
                    source_state_key=current_loc_key, goal_state_key=goal_key
                )
                if path is not None:
                    paths.append(path)
            if len(paths) == 0:
                return 0, False

            shortest_path_ind = int(np.argmin([len(p) for p in paths]))

            if len(paths[shortest_path_ind]) == 1:
                warnings.warn(
                    "Shortest path computations suggest we are at the target but episode does not think so."
                )
                return 0, False

            next_key_on_shortest_path = paths[shortest_path_ind][1]
            return (
                self.action_names().index(
                    self.env.action_transitioning_between_keys(
                        current_loc_key, next_key_on_shortest_path
                    )
                ),
                True,
            )


class ObjectNavTheRobotProjectTask(ObjectNavTask):
    def __init__(
        self,
        env: AI2ThorEnvironment,
        sensors: List[Sensor],
        task_info: Dict[str, Any],
        max_steps: int,
        reward_structure_configs: Dict[str, Any],
        **kwargs
    ) -> None:
        super().__init__(env, sensors, task_info, max_steps)
        self.reward_structure_configs = reward_structure_configs
        self.is_robot = False
        self.scene_graphs = self.load_scene_graphs(
            self.reward_structure_configs["scene_graphs_path"]
        )

    @staticmethod
    def nn(opoints, osource):
        points = opoints[:, [0, 2]]
        source = osource[[0, 2], :]
        dist2 = np.sum(points * points, axis=-1).reshape(-1, 1)
        dist2 += np.dot(source.transpose(), source)
        dist2 -= 2 * np.dot(points, source)
        minpos = np.argmin(dist2)
        return opoints[minpos, :], dist2[minpos]

    @staticmethod
    def load_scene_graphs(path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        return data

    def grid_location(self):
        xg = self.scene_graphs[self.env.scene_name]["grid_points"]

        # xn = self.environment.current_agent_attributes()["xyz"]
        pos = self.env.get_agent_location()
        xn = [pos[k] for k in ["x", "y", "z"]]
        xn = np.array(xn).reshape((3, 1))
        agent_pose, _ = self.nn(xg, xn)

        return agent_pose

    def target_visible(self):
        boxes = self.env.last_event.class_detections2D
        if self.task_info["object_type"] in boxes:
            for box in boxes[self.task_info["object_type"]]:
                if (box[2] - box[0]) * (
                    box[3] - box[1]
                ) < self.reward_structure_configs["min_target_area"]:
                    continue
                return True
        return False

    def judge(self) -> float:
        """ Judge the last event. """
        # immediate reward
        reward = self.reward_structure_configs["step_penalty"]

        new_dist = self.dist_to_target()

        if self.reward_structure_configs["shaping_type"] == "dist":
            reward += self.get_reward_dist_shaping()
        elif self.reward_structure_configs["shaping_type"] == "delta_dist":
            reward += self.get_reward_delta_shaping(new_dist)
        elif self.reward_structure_configs["shaping_type"] == "explore_then_dist":
            if self.target_visible():
                reward += self.get_reward_dist_shaping()
            else:
                reward += self.get_reward_exploration()
        elif self.reward_structure_configs["shaping_type"] == "explore_then_delta_dist":
            if self.target_visible():
                reward += self.get_reward_delta_shaping(new_dist)
            else:
                reward += self.get_reward_exploration()
        elif self.reward_structure_configs["shaping_type"] == "no":
            reward += 0
        else:
            raise ValueError("Incorrect reward shaping parameter")

        self.last_dist = new_dist

        if self._took_end_action:
            reward += (
                self.reward_structure_configs["goal_success_reward"]
                if self._success
                else -1.0
            )

        return float(reward)

    def get_reward_delta_shaping(self, new_dist):
        if new_dist < self.last_dist:
            reward = self.reward_structure_configs["delta_dist_penalty_closer"]
        elif new_dist == self.last_dist:
            reward = self.reward_structure_configs["delta_dist_penalty_same"]
        else:
            reward = self.reward_structure_configs["delta_dist_penalty_further"]
        return reward

    def dist_to_explored(self):
        agent_pose = self.grid_location()
        _, min_dist2 = self.nn(self.visited_locations, agent_pose.reshape(3, 1))
        return min_dist2, agent_pose

    @staticmethod
    def object_is_target(object, target, is_id=False):
        return (is_id and target == object["objectId"]) or (
            not is_id and target == object["objectType"]
        )

    def istarget(self, object):
        return self.object_is_target(
            object,
            self.task_info["object_type"],
            self.reward_structure_configs["target_is_id"],
        )

    @staticmethod
    def key_for_point(point):
        return "%0.3f|%0.3f" % (point[0], point[2])

    def dist_to_target(self):
        if "shaping_type" not in self.reward_structure_configs:
            return 0

        if self.is_robot:
            return 0

        if (
            self.reward_structure_configs["shaping_type"] == "dist"
            and self.reward_structure_configs["dist_penalty_scale"] == 0
        ):
            return 0

        xg = self.scene_graphs[self.env.scene_name]["grid_points"]

        xt = [
            obj["position"]
            for obj in self.env.last_event.metadata["objects"]
            if self.istarget(obj)
        ]
        xt = np.array([xt[0][axis] for axis in ["x", "y", "z"]]).reshape((3, 1))
        target_pose, _ = self.nn(xg, xt)
        # logger.info('source %s -> %s' % (xt, target_pose))

        # xn = self.env.current_agent_attributes()["xyz"]
        pos = self.env.get_agent_location()
        xn = [pos[k] for k in ["x", "y", "z"]]
        xn = np.array(xn).reshape((3, 1))
        agent_pose, _ = self.nn(xg, xn)
        # logger.info('agent %s -> %s' % (xn, agent_pose))

        dist = len(
            nx.shortest_path(
                self.scene_graphs[self.env.scene_name]["graph"],
                self.key_for_point(agent_pose),
                self.key_for_point(target_pose),
            )
        )  # TODO could be cached
        # dist = self.all_dists[self.key_for_point(agent_pose)]

        return dist

    def get_reward_dist_shaping(self):
        reward = (
            -self.dist_to_target() * self.reward_structure_configs["dist_penalty_scale"]
        )
        return reward

    def get_reward_exploration(self):
        reward = 0
        if self.visited_locations is None:
            self.visited_locations = self.grid_location().reshape(1, 3)
        else:
            min_dist, grid_loc = self.dist_to_explored()
            if (
                min_dist
                < (
                    self.reward_structure_configs["grid_size"]
                    * self.reward_structure_configs["min_dist_factor"]
                )
                ** 2
            ):
                reward = self.reward_structure_configs["revisit_penalty"]
            else:
                reward = self.reward_structure_configs["exploration_reward"]
                self.visited_locations = np.concatenate(
                    (self.visited_locations, grid_loc.reshape((1, 3))), axis=0
                )
        return reward
