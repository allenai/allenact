"""Task Definions for the task of ArmPointNav."""

import copy
from typing import Dict, Tuple, List, Any, Optional

import gym
import numpy as np

from allenact.base_abstractions.misc import RLStepResult
from allenact.base_abstractions.sensor import Sensor
from allenact.base_abstractions.task import Task
from allenact_plugins.manipulathor_plugin.armpointnav_constants import (
    MOVE_ARM_CONSTANT,
    DISTANCE_EPS,
)
from allenact_plugins.manipulathor_plugin.manipulathor_constants import (
    MOVE_ARM_HEIGHT_P,
    MOVE_ARM_HEIGHT_M,
    MOVE_ARM_X_P,
    MOVE_ARM_X_M,
    MOVE_ARM_Y_P,
    MOVE_ARM_Y_M,
    MOVE_ARM_Z_P,
    MOVE_ARM_Z_M,
    ROTATE_WRIST_PITCH_P,
    ROTATE_WRIST_PITCH_M,
    ROTATE_WRIST_YAW_P,
    ROTATE_WRIST_YAW_M,
    ROTATE_ELBOW_P,
    ROTATE_ELBOW_M,
    LOOK_UP,
    LOOK_DOWN,
    MOVE_AHEAD,
    ROTATE_RIGHT,
    ROTATE_LEFT,
    PICKUP,
    DONE,
)
from allenact_plugins.manipulathor_plugin.manipulathor_environment import (
    ManipulaTHOREnvironment,
    position_distance,
)
from allenact_plugins.manipulathor_plugin.manipulathor_viz import LoggerVisualizer


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

    # New commit of AI2THOR has some issue that the objects will vibrate a bit
    # without any external force. To eliminate the vibration effect, we have to
    # introduce _vibration_dist_dict when checking the disturbance, from an external csv file.
    # By default it is None, i.e. we assume there is no vibration.

    _vibration_dist_dict: Optional[Dict] = None

    def __init__(
        self,
        env: ManipulaTHOREnvironment,
        sensors: List[Sensor],
        task_info: Dict[str, Any],
        max_steps: int,
        visualizers: Optional[List[LoggerVisualizer]] = None,
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
        self.visualizers = visualizers if visualizers is not None else []
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
        self.initial_object_locations = self.env.get_current_object_locations()

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
            visualizer.log(self.env)

    def visualize(self, action_str):

        for vizualizer in self.visualizers:
            vizualizer.log(self.env, action_str)

    def finish_visualizer(self):

        for visualizer in self.visualizers:
            visualizer.finish_episode(self.env, self, self.task_info)

    def finish_visualizer_metrics(self, metric_results):

        for visualizer in self.visualizers:
            visualizer.finish_episode_metrics(self, self.task_info, metric_results)

    def render(self, mode: str = "rgb", *args, **kwargs) -> np.ndarray:
        assert mode == "rgb", "only rgb rendering is implemented"
        return self.env.current_frame

    def calc_action_stat_metrics(self) -> Dict[str, Any]:
        action_stat = {"action_stat/" + action_str: 0.0 for action_str in self._actions}
        action_success_stat = {
            "action_success/" + action_str: 0.0 for action_str in self._actions
        }
        action_success_stat["action_success/total"] = 0.0

        seq_len = len(self.action_sequence_and_success)
        for (action_name, action_success) in self.action_sequence_and_success:
            action_stat["action_stat/" + action_name] += 1.0
            action_success_stat[
                "action_success/{}".format(action_name)
            ] += action_success
            action_success_stat["action_success/total"] += action_success

        action_success_stat["action_success/total"] /= seq_len

        for action_name in self._actions:
            action_success_stat["action_success/{}".format(action_name)] /= max(
                action_stat["action_stat/" + action_name], 1.0
            )
            action_stat["action_stat/" + action_name] /= seq_len

        result = {**action_stat, **action_success_stat}

        return result

    def metrics(self) -> Dict[str, Any]:
        result = super(AbstractPickUpDropOffTask, self).metrics()

        if self.is_done():
            result = {**result, **self.calc_action_stat_metrics()}

            # 1. goal object metrics
            final_obj_distance_from_goal = self.obj_distance_from_goal()
            result[
                "average/final_obj_distance_from_goal"
            ] = final_obj_distance_from_goal
            final_arm_distance_from_obj = self.arm_distance_from_obj()
            result["average/final_arm_distance_from_obj"] = final_arm_distance_from_obj

            final_obj_pickup = 1 if self.object_picked_up else 0
            result["average/final_obj_pickup"] = final_obj_pickup

            original_distance = self.get_original_object_distance() + DISTANCE_EPS
            result["average/original_distance"] = original_distance

            # this ratio can be more than 1
            if self.object_picked_up:
                ratio_distance_left = final_obj_distance_from_goal / original_distance
                result["average/ratio_distance_left"] = ratio_distance_left
                result["average/eplen_pickup"] = self.eplen_pickup

            # 2. disturbance with other objects
            current_object_locations = self.env.get_current_object_locations()
            objects_moved = self.env.get_objects_moved(
                self.initial_object_locations,
                current_object_locations,
                self.task_info["objectId"],
                self._vibration_dist_dict,
            )
            result["disturbance/objects_moved_num"] = len(objects_moved)

            # 3. conditioned on success
            if self._success:
                result["average/eplen_success"] = result["ep_length"]
                result["average/success_wo_disturb"] = len(objects_moved) == 0

            else:
                result["average/success_wo_disturb"] = 0.0

            result["success"] = self._success

            self.finish_visualizer_metrics(result)
            self.finish_visualizer()
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

    def __init__(
        self,
        env: ManipulaTHOREnvironment,
        sensors: List[Sensor],
        task_info: Dict[str, Any],
        max_steps: int,
        visualizers: Optional[List[LoggerVisualizer]] = None,
        **kwargs
    ) -> None:
        super().__init__(
            env=env,
            sensors=sensors,
            task_info=task_info,
            max_steps=max_steps,
            visualizers=visualizers,
            **kwargs
        )
        self.cumulated_disturb_distance_all = 0.0
        self.cumulated_disturb_distance_visible = 0.0
        # NOTE: visible distance can be negative, no determinitic relation with
        #   all distance
        self.previous_object_locations = copy.deepcopy(self.initial_object_locations)
        self.current_penalized_distance = 0.0  # used in Sensor for auxiliary task

    def metrics(self) -> Dict[str, Any]:
        result = super(ArmPointNavTask, self).metrics()

        if self.is_done():
            # add disturbance distance metrics
            result[
                "disturbance/objects_moved_distance"
            ] = self.cumulated_disturb_distance_all
            result[
                "disturbance/objects_moved_distance_vis"
            ] = self.cumulated_disturb_distance_visible

        return result

    def visualize(self, **kwargs):

        for vizualizer in self.visualizers:
            vizualizer.log(self.env, **kwargs)

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

        # add disturbance cost
        ## here we measure disturbance by the sum of moving distance of all objects
        ## note that collided object may move for a while wo external force due to inertia
        ## and we may also consider mass
        current_object_locations = self.env.get_current_object_locations()

        disturb_distance_visible = self.env.get_objects_move_distance(
            initial_object_locations=self.initial_object_locations,
            previous_object_locations=self.previous_object_locations,
            current_object_locations=current_object_locations,
            target_object_id=self.task_info["objectId"],
            only_visible=True,
            thres_dict=self._vibration_dist_dict,
        )
        disturb_distance_all = self.env.get_objects_move_distance(
            initial_object_locations=self.initial_object_locations,
            previous_object_locations=self.previous_object_locations,
            current_object_locations=current_object_locations,
            target_object_id=self.task_info["objectId"],
            only_visible=False,
            thres_dict=self._vibration_dist_dict,
        )

        self.cumulated_disturb_distance_all += disturb_distance_all
        self.cumulated_disturb_distance_visible += disturb_distance_visible

        penalized_distance = (
            disturb_distance_visible
            if self.reward_configs["disturb_visible"]
            else disturb_distance_all
        )
        reward += self.reward_configs["disturb_penalty"] * penalized_distance
        self.current_penalized_distance = penalized_distance

        self.previous_object_locations = current_object_locations

        self.visualize(
            action_str=self._last_action_str,
            disturbance_str=str(round(penalized_distance, 4)),
        )

        return float(reward)


class RotateArmPointNavTask(ArmPointNavTask):
    _actions = (
        MOVE_ARM_HEIGHT_P,
        MOVE_ARM_HEIGHT_M,
        MOVE_ARM_X_P,
        MOVE_ARM_X_M,
        MOVE_ARM_Y_P,
        MOVE_ARM_Y_M,
        MOVE_ARM_Z_P,
        MOVE_ARM_Z_M,
        ROTATE_WRIST_PITCH_P,
        ROTATE_WRIST_PITCH_M,
        ROTATE_WRIST_YAW_P,
        ROTATE_WRIST_YAW_M,
        ROTATE_ELBOW_P,
        ROTATE_ELBOW_M,
        MOVE_AHEAD,
        ROTATE_RIGHT,
        ROTATE_LEFT,
        PICKUP,
        DONE,
    )


class CamRotateArmPointNavTask(ArmPointNavTask):
    _actions = (
        MOVE_ARM_HEIGHT_P,
        MOVE_ARM_HEIGHT_M,
        MOVE_ARM_X_P,
        MOVE_ARM_X_M,
        MOVE_ARM_Y_P,
        MOVE_ARM_Y_M,
        MOVE_ARM_Z_P,
        MOVE_ARM_Z_M,
        ROTATE_WRIST_PITCH_P,
        ROTATE_WRIST_PITCH_M,
        ROTATE_WRIST_YAW_P,
        ROTATE_WRIST_YAW_M,
        ROTATE_ELBOW_P,
        ROTATE_ELBOW_M,
        LOOK_UP,
        LOOK_DOWN,
        MOVE_AHEAD,
        ROTATE_RIGHT,
        ROTATE_LEFT,
        PICKUP,
        DONE,
    )


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
            if (
                object_id
                in self.env.controller.last_event.metadata["arm"]["pickupableObjects"]
            ):
                self.env.step(dict(action="PickupObject"))
                #  we are doing an additional pass here, label is not right and if we fail we will do it twice
                object_inventory = self.env.controller.last_event.metadata["arm"][
                    "heldObjects"
                ]
                if len(object_inventory) > 0 and object_id not in object_inventory:
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
