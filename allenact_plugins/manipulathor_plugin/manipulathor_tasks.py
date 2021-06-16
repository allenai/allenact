"""Task Definions for the task of ArmPointNav"""

from typing import Dict, Tuple, List, Any, Optional

import gym
import numpy as np
from allenact.base_abstractions.misc import RLStepResult
from allenact.base_abstractions.sensor import Sensor
from allenact.base_abstractions.task import Task

from allenact_plugins.manipulathor_plugin.manipulathor_constants import (
    MOVE_ARM_CONSTANT,
    MOVE_ARM_HEIGHT_P,
    MOVE_ARM_HEIGHT_M,
    MOVE_ARM_X_P,
    MOVE_ARM_X_M,
    MOVE_ARM_Y_P,
    MOVE_ARM_Y_M,
    MOVE_ARM_Z_P,
    MOVE_ARM_Z_M,
    MOVE_AHEAD,
    ROTATE_RIGHT,
    ROTATE_LEFT,
    PICKUP,
    DONE,
)
from allenact_plugins.manipulathor_plugin.manipulathor_environment import ManipulaTHOREnvironment
from allenact_plugins.manipulathor_plugin.manipulathor_viz import LoggerVisualizer


def position_distance(s1, s2):
    position1 = s1["position"]
    position2 = s2["position"]
    return (
        (position1["x"] - position2["x"]) ** 2
        + (position1["y"] - position2["y"]) ** 2
        + (position1["z"] - position2["z"]) ** 2
    ) ** 0.5


class AbstractPickUpDropOffTask(Task[ManipulaTHOREnvironment]):

    _actions = (
        MOVE_ARM_HEIGHT_P,
        MOVE_ARM_HEIGHT_M,
        MOVE_ARM_X_P,
        MOVE_ARM_X_M,
        MOVE_ARM_Y_P,
        MOVE_ARM_Y_M,
        MOVE_ARM_Z_P,
        MOVE_ARM_Z_M,
        MOVE_AHEAD,
        ROTATE_RIGHT,
        ROTATE_LEFT,
    )

    def __init__(
        self,
        env: ManipulaTHOREnvironment,
        sensors: List[Sensor],
        task_info: Dict[str, Any],
        max_steps: int,
        visualizers: List[LoggerVisualizer] = [],
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
        self.visualizers = visualizers
        self.start_visualize()
        self.action_sequence_and_success = []
        self._took_end_action: bool = False
        self._success: Optional[bool] = False
        self._subsampled_locations_from_which_obj_visible: Optional[
            List[Tuple[float, float, int, int]]
        ] = None

        # in allenact initialization is with 0.2
        self.last_obj_to_goal_distance = None
        self.last_arm_to_obj_distance = None
        self.object_picked_up = False
        self.got_reward_for_pickup = False
        self.reward_configs = kwargs["reward_configs"]
        self.initial_object_metadata = self.env.get_current_object_locations()

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

    def obj_state_aproximity(self, s1, s2):
        # KIANA ignore rotation for now
        position1 = s1["position"]
        position2 = s2["position"]
        eps = MOVE_ARM_CONSTANT * 2
        return (
            abs(position1["x"] - position2["x"]) < eps
            and abs(position1["y"] - position2["y"]) < eps
            and abs(position1["z"] - position2["z"]) < eps
        )

    def start_visualize(self):
        for visualizer in self.visualizers:
            if not visualizer.is_empty():
                print("OH NO VISUALIZER WAS NOT EMPTY")
                visualizer.finish_episode(self.env, self, self.task_info)
                visualizer.finish_episode_metrics(self, self.task_info, None)
            visualizer.log(self.env, "")

    def visualize(self, action_str):

        for vizualizer in self.visualizers:
            vizualizer.log(self.env, action_str)

    def finish_visualizer(self, episode_success):

        for visualizer in self.visualizers:
            visualizer.finish_episode(self.env, self, self.task_info)

    def finish_visualizer_metrics(self, metric_results):

        for visualizer in self.visualizers:
            visualizer.finish_episode_metrics(self, self.task_info, metric_results)

    def render(self, mode: str = "rgb", *args, **kwargs) -> np.ndarray:
        assert mode == "rgb", "only rgb rendering is implemented"
        return self.env.current_frame

    def calc_action_stat_metrics(self) -> Dict[str, Any]:
        action_stat = {
            "metric/action_stat/" + action_str: 0.0 for action_str in self._actions
        }
        action_success_stat = {
            "metric/action_success/" + action_str: 0.0 for action_str in self._actions
        }
        action_success_stat["metric/action_success/total"] = 0.0

        seq_len = len(self.action_sequence_and_success)
        for (action_name, action_success) in self.action_sequence_and_success:
            action_stat["metric/action_stat/" + action_name] += 1.0
            action_success_stat[
                "metric/action_success/{}".format(action_name)
            ] += action_success
            action_success_stat["metric/action_success/total"] += action_success

        action_success_stat["metric/action_success/total"] /= seq_len

        for action_name in self._actions:
            action_success_stat[
                "metric/" + "action_success/{}".format(action_name)
            ] /= (action_stat["metric/action_stat/" + action_name] + 0.000001)
            action_stat["metric/action_stat/" + action_name] /= seq_len

        succ = [v for v in action_success_stat.values()]
        sum(succ) / len(succ)
        result = {**action_stat, **action_success_stat}

        return result

    def metrics(self) -> Dict[str, Any]:
        result = super(AbstractPickUpDropOffTask, self).metrics()
        if self.is_done():
            result = {**result, **self.calc_action_stat_metrics()}
            final_obj_distance_from_goal = self.obj_distance_from_goal()
            result[
                "metric/average/final_obj_distance_from_goal"
            ] = final_obj_distance_from_goal
            final_arm_distance_from_obj = self.arm_distance_from_obj()
            result[
                "metric/average/final_arm_distance_from_obj"
            ] = final_arm_distance_from_obj
            final_obj_pickup = 1 if self.object_picked_up else 0
            result["metric/average/final_obj_pickup"] = final_obj_pickup

            original_distance = self.get_original_object_distance()
            result["metric/average/original_distance"] = original_distance

            # this ratio can be more than 1?
            if self.object_picked_up:
                ratio_distance_left = final_obj_distance_from_goal / original_distance
                result["metric/average/ratio_distance_left"] = ratio_distance_left
                result["metric/average/eplen_pickup"] = self.eplen_pickup

            if self._success:
                result["metric/average/eplen_success"] = result["ep_length"]
                # put back this is not the reason for being slow
                objects_moved = self.env.get_objects_moved(self.initial_object_metadata)
                # Unnecessary, this is definitely happening objects_moved.remove(self.task_info['object_id'])
                result["metric/average/number_of_unwanted_moved_objects"] = (
                    len(objects_moved) - 1
                )
                result["metric/average/success_wo_disturb"] = (
                    len(objects_moved) == 1
                )  # multiply this by the successrate

            result["success"] = self._success

            self.finish_visualizer_metrics(result)
            self.finish_visualizer(self._success)
            self.action_sequence_and_success = []

        return result

    def _step(self, action: int) -> RLStepResult:
        raise Exception("Not implemented")

    def arm_distance_from_obj(self):
        goal_obj_id = self.task_info["objectId"]
        object_info = self.env.get_object_by_id(goal_obj_id)
        hand_state = self.env.get_absolute_hand_state()
        return position_distance(object_info, hand_state)

    def obj_distance_from_goal(self):
        goal_obj_id = self.task_info["objectId"]
        object_info = self.env.get_object_by_id(goal_obj_id)
        goal_state = self.task_info["target_location"]
        return position_distance(object_info, goal_state)

    def get_original_object_distance(self):
        goal_obj_id = self.task_info["objectId"]
        s_init = dict(position=self.task_info["source_location"]["object_location"])
        current_location = self.env.get_object_by_id(goal_obj_id)

        original_object_distance = position_distance(s_init, current_location)
        return original_object_distance

    def judge(self) -> float:
        """Compute the reward after having taken a step."""
        raise Exception("Not implemented")


class ArmPointNavTask(AbstractPickUpDropOffTask):
    _actions = (
        MOVE_ARM_HEIGHT_P,
        MOVE_ARM_HEIGHT_M,
        MOVE_ARM_X_P,
        MOVE_ARM_X_M,
        MOVE_ARM_Y_P,
        MOVE_ARM_Y_M,
        MOVE_ARM_Z_P,
        MOVE_ARM_Z_M,
        MOVE_AHEAD,
        ROTATE_RIGHT,
        ROTATE_LEFT,
        PICKUP,
        DONE,
    )

    def _step(self, action: int) -> RLStepResult:

        action_str = self.class_action_names()[action]

        self._last_action_str = action_str
        action_dict = {"action": action_str}
        object_id = self.task_info["objectId"]
        if action_str == PICKUP:
            action_dict = {**action_dict, "object_id": object_id}
        self.env.step(action_dict)
        self.last_action_success = self.env.last_action_success

        last_action_name = self._last_action_str
        last_action_success = float(self.last_action_success)
        self.action_sequence_and_success.append((last_action_name, last_action_success))
        self.visualize(last_action_name)

        # If the object has not been picked up yet and it was picked up in the previous step update parameters to integrate it into reward
        if not self.object_picked_up:

            if self.env.is_object_at_low_level_hand(object_id):
                self.object_picked_up = True
                self.eplen_pickup = (
                    self._num_steps_taken + 1
                )  # plus one because this step has not been counted yet

        if action_str == DONE:
            self._took_end_action = True
            object_state = self.env.get_object_by_id(object_id)
            goal_state = self.task_info["target_location"]
            goal_achieved = self.object_picked_up and self.obj_state_aproximity(
                object_state, goal_state
            )
            self.last_action_success = goal_achieved
            self._success = goal_achieved

        step_result = RLStepResult(
            observation=self.get_observations(),
            reward=self.judge(),
            done=self.is_done(),
            info={"last_action_success": self.last_action_success},
        )
        return step_result

    def judge(self) -> float:
        """Compute the reward after having taken a step."""
        reward = self.reward_configs["step_penalty"]

        if not self.last_action_success or (
            self._last_action_str == PICKUP and not self.object_picked_up
        ):
            reward += self.reward_configs["failed_action_penalty"]

        if self._took_end_action:
            reward += (
                self.reward_configs["goal_success_reward"]
                if self._success
                else self.reward_configs["failed_stop_reward"]
            )

        # increase reward if object pickup and only do it once
        if not self.got_reward_for_pickup and self.object_picked_up:
            reward += self.reward_configs["pickup_success_reward"]
            self.got_reward_for_pickup = True

        current_obj_to_arm_distance = self.arm_distance_from_obj()
        if self.last_arm_to_obj_distance is None:
            delta_arm_to_obj_distance_reward = 0
        else:
            delta_arm_to_obj_distance_reward = (
                self.last_arm_to_obj_distance - current_obj_to_arm_distance
            )
        self.last_arm_to_obj_distance = current_obj_to_arm_distance
        reward += delta_arm_to_obj_distance_reward

        current_obj_to_goal_distance = self.obj_distance_from_goal()
        if self.last_obj_to_goal_distance is None:
            delta_obj_to_goal_distance_reward = 0
        else:
            delta_obj_to_goal_distance_reward = (
                self.last_obj_to_goal_distance - current_obj_to_goal_distance
            )
        self.last_obj_to_goal_distance = current_obj_to_goal_distance
        reward += delta_obj_to_goal_distance_reward

        # add collision cost, maybe distance to goal objective,...

        return float(reward)

class EasyArmPointNavTask(ArmPointNavTask):
    _actions = (
        MOVE_ARM_HEIGHT_P,
        MOVE_ARM_HEIGHT_M,
        MOVE_ARM_X_P,
        MOVE_ARM_X_M,
        MOVE_ARM_Y_P,
        MOVE_ARM_Y_M,
        MOVE_ARM_Z_P,
        MOVE_ARM_Z_M,
        MOVE_AHEAD,
        ROTATE_RIGHT,
        ROTATE_LEFT,
        # PICKUP,
        # DONE,
    )
    def _step(self, action: int) -> RLStepResult:

        action_str = self.class_action_names()[action]

        self._last_action_str = action_str
        action_dict = {"action": action_str}
        object_id = self.task_info["objectId"]
        if action_str == PICKUP:
            action_dict = {**action_dict, "object_id": object_id}
        self.env.step(action_dict)
        self.last_action_success = self.env.last_action_success

        last_action_name = self._last_action_str
        last_action_success = float(self.last_action_success)
        self.action_sequence_and_success.append((last_action_name, last_action_success))
        self.visualize(last_action_name)

        # If the object has not been picked up yet and it was picked up in the previous step update parameters to integrate it into reward
        if not self.object_picked_up:
            if object_id in self.env.controller.last_event.metadata['arm']['pickupableObjects']:
                self.env.step(dict(action="PickupObject"))
                #  we are doing an additional pass here, label is not right and if we fail we will do it twice
                object_inventory = self.env.controller.last_event.metadata["arm"][
                    "heldObjects"
                ]
                if (
                        len(object_inventory) > 0
                        and object_id not in object_inventory
                ):
                    self.env.step(dict(action="ReleaseObject"))

            if self.env.is_object_at_low_level_hand(object_id):
                self.object_picked_up = True
                self.eplen_pickup = (
                        self._num_steps_taken + 1
                )  # plus one because this step has not been counted yet

        if self.object_picked_up:

            object_state = self.env.get_object_by_id(object_id)
            goal_state = self.task_info["target_location"]
            goal_achieved = self.object_picked_up and self.obj_state_aproximity(
                object_state, goal_state
            )
            if goal_achieved:
                self._took_end_action = True
                self.last_action_success = goal_achieved
                self._success = goal_achieved

        step_result = RLStepResult(
            observation=self.get_observations(),
            reward=self.judge(),
            done=self.is_done(),
            info={"last_action_success": self.last_action_success},
        )
        return step_result

    # def judge(self) -> float: Seems like we are fine on this