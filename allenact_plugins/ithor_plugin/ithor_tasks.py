import random
from typing import Dict, Tuple, List, Any, Optional, Union, Sequence, cast

import gym
import numpy as np

from allenact.base_abstractions.misc import RLStepResult
from allenact.base_abstractions.sensor import Sensor
from allenact.base_abstractions.task import Task
from allenact.utils.system import get_logger
from allenact_plugins.ithor_plugin.ithor_constants import (
    MOVE_AHEAD,
    ROTATE_LEFT,
    ROTATE_RIGHT,
    LOOK_DOWN,
    LOOK_UP,
    END,
)

from allenact_plugins.ithor_plugin.ithor_environment import IThorEnvironment
from allenact_plugins.ithor_plugin.ithor_util import round_to_factor


class ObjectNaviThorGridTask(Task[IThorEnvironment]):
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
        env: IThorEnvironment,
        sensors: List[Sensor],
        task_info: Dict[str, Any],
        max_steps: int,
        **kwargs,
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

        self.task_info["followed_path"] = [self.env.get_agent_location()]
        self.task_info["action_names"] = self.class_action_names()

    @property
    def action_space(self):
        return gym.spaces.Discrete(len(self._actions))

    def reached_terminal_state(self) -> bool:
        return self._took_end_action

    @classmethod
    def class_action_names(cls, **kwargs) -> Tuple[str, ...]:
        return cls._actions

    def close(self) -> None:
        self.env.stop()

    def _step(self, action: Union[int, Sequence[int]]) -> RLStepResult:
        assert isinstance(action, int)
        action = cast(int, action)

        action_str = self.class_action_names()[action]

        if action_str == END:
            self._took_end_action = True
            self._success = self.is_goal_object_visible()
            self.last_action_success = self._success
        else:
            self.env.step({"action": action_str})
            self.last_action_success = self.env.last_action_success

            if (
                not self.last_action_success
            ) and self._CACHED_LOCATIONS_FROM_WHICH_OBJECT_IS_VISIBLE is not None:
                self.env.update_graph_with_failed_action(failed_action=action_str)

            self.task_info["followed_path"].append(self.env.get_agent_location())

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

    def is_goal_object_visible(self) -> bool:
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
            return {
                "success": self._success,
                **super(ObjectNaviThorGridTask, self).metrics(),
            }

    def query_expert(self, **kwargs) -> Tuple[int, bool]:
        target = self.task_info["object_type"]

        if self.is_goal_object_visible():
            return self.class_action_names().index(END), True
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
                get_logger().warning(
                    "Shortest path computations suggest we are at the target but episode does not think so."
                )
                return 0, False

            next_key_on_shortest_path = paths[shortest_path_ind][1]
            return (
                self.class_action_names().index(
                    self.env.action_transitioning_between_keys(
                        current_loc_key, next_key_on_shortest_path
                    )
                ),
                True,
            )
