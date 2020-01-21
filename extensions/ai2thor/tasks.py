import warnings
import random
from typing import Tuple, List, Dict, Any, Optional

import gym
import numpy as np

from extensions.ai2thor.constants import (
    MOVE_AHEAD,
    ROTATE_LEFT,
    ROTATE_RIGHT,
    LOOK_DOWN,
    LOOK_UP,
    END,
)
from extensions.ai2thor.environment import AI2ThorEnvironment
from extensions.ai2thor.misc_util import round_to_factor
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

    def _step(self, action: int) -> RLStepResult:
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
    _actions = (MOVE_AHEAD, ROTATE_LEFT, ROTATE_RIGHT, LOOK_DOWN, LOOK_UP, END)

    _CACHED_LOCATIONS_FROM_WHICH_OBJECT_IS_VISIBLE: Dict[
        Tuple[str, str], Tuple[float, float, int, int]
    ] = {}

    def __init__(
        self,
        env: AI2ThorEnvironment,
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
        return any(
            o["objectType"] == self.task_info["object_type"]
            for o in self.env.visible_objects()
        )

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
        target = self.task_info["object_type"]

        if self._is_goal_object_visible():
            return self.action_names().index(END), True
        else:
            key = (self.env.scene_name, target)
            if self._subsampled_locations_from_which_obj_visible is None:
                if key not in self._CACHED_LOCATIONS_FROM_WHICH_OBJECT_IS_VISIBLE:
                    obj_ids = []
                    obj_ids.extend(
                        o["objectId"]
                        for o in self.env.last_event.metadata["objects"]
                        if o["objectType"] == target
                    )

                    assert len(obj_ids) != 0, "No objects to get an expert path to."

                    locations_from_which_object_is_visible = []
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
