"""Task Samplers for the task of ArmPointNav"""
import json
import random
from typing import List, Dict, Optional, Any, Union

import gym
from allenact.base_abstractions.sensor import Sensor
from allenact.base_abstractions.task import Task
from allenact.base_abstractions.task import TaskSampler
from allenact.utils.experiment_utils import set_deterministic_cudnn, set_seed
from allenact_plugins.manipulathor_plugin.arm_calulcation_util import initialize_arm
from allenact_plugins.manipulathor_plugin.manipulathor_constants import transport_wrapper
from allenact_plugins.manipulathor_plugin.manipulathor_environment import ManipulaTHOREnvironment
from allenact_plugins.manipulathor_plugin.manipulathor_tasks import AbstractPickUpDropOffTask, ArmPointNavTask, EasyArmPointNavTask
from allenact_plugins.manipulathor_plugin.manipulathor_viz import ImageVisualizer


class AbstractMidLevelArmTaskSampler(TaskSampler):

    _TASK_TYPE = Task

    def __init__(
        self,
        scenes: List[str],
        sensors: List[Sensor],
        max_steps: int,
        env_args: Dict[str, Any],
        action_space: gym.Space,
        rewards_config: Dict,
        objects: List[str],
        scene_period: Optional[Union[int, str]] = None,
        max_tasks: Optional[int] = None,
        seed: Optional[int] = None,
        deterministic_cudnn: bool = False,
        fixed_tasks: Optional[List[Dict[str, Any]]] = None,
        visualizers: List[LoggerVisualizer] = [],
        *args,
        **kwargs
    ) -> None:
        self.rewards_config = rewards_config
        self.env_args = env_args
        self.scenes = scenes
        self.grid_size = 0.25
        self.env: Optional[ManipulaTHOREnvironment] = None
        self.sensors = sensors
        self.max_steps = max_steps
        self._action_space = action_space
        self.objects = objects

        self.scene_counter: Optional[int] = None
        self.scene_order: Optional[List[str]] = None
        self.scene_id: Optional[int] = None
        self.scene_period: Optional[
            Union[str, int]
        ] = scene_period  # default makes a random choice
        self.max_tasks: Optional[int] = None
        self.reset_tasks = max_tasks

        self._last_sampled_task: Optional[Task] = None

        self.seed: Optional[int] = None
        self.set_seed(seed)

        if deterministic_cudnn:
            set_deterministic_cudnn()

        self.reset()
        self.visualizers = visualizers
        self.sampler_mode = kwargs["sampler_mode"]
        self.cap_training = kwargs["cap_training"]

    def _create_environment(self, **kwargs) -> ManipulaTHOREnvironment:
        env = ManipulaTHOREnvironment(
            make_agents_visible=False,
            object_open_speed=0.05,
            env_args=self.env_args,
        )

        return env

    @property
    def last_sampled_task(self) -> Optional[Task]:
        return self._last_sampled_task

    def close(self) -> None:
        if self.env is not None:
            self.env.stop()

    @property
    def all_observation_spaces_equal(self) -> bool:
        """Check if observation spaces equal.

        # Returns

        True if all Tasks that can be sampled by this sampler have the
            same observation space. Otherwise False.
        """
        return True

    def reset(self):
        self.scene_counter = 0
        self.scene_order = list(range(len(self.scenes)))
        random.shuffle(self.scene_order)
        self.scene_id = 0
        self.sampler_index = 0

        self.max_tasks = self.reset_tasks

    def set_seed(self, seed: int):
        self.seed = seed
        if seed is not None:
            set_seed(seed)


class SimpleArmPointNavGeneralSampler(AbstractMidLevelArmTaskSampler):

    _TASK_TYPE = AbstractPickUpDropOffTask

    def __init__(self, **kwargs) -> None:

        super(SimpleArmPointNavGeneralSampler, self).__init__(**kwargs)
        self.all_possible_points = []
        for scene in self.scenes:
            for object in self.objects:
                valid_position_adr = "datasets/apnd-dataset/valid_object_positions/valid_{}_positions_in_{}.json".format(
                    object, scene
                )
                try:
                    with open(valid_position_adr) as f:
                        data_points = json.load(f)
                except Exception:
                    print("Failed to load", valid_position_adr)
                    continue
                visible_data = [
                    data for data in data_points[scene] if data["visibility"]
                ]
                self.all_possible_points += visible_data

        self.countertop_object_to_data_id = self.calc_possible_trajectories(
            self.all_possible_points
        )

        scene_names = set(
            [
                self.all_possible_points[counter[0]]["scene_name"]
                for counter in self.countertop_object_to_data_id.values()
                if len(counter) > 1
            ]
        )

        if len(set(scene_names)) < len(self.scenes):
            print("Not all scenes appear")

        print(
            "Len dataset",
            len(self.all_possible_points),
            "total_remained",
            sum([len(v) for v in self.countertop_object_to_data_id.values()]),
        )

        if (
            self.sampler_mode != "train"
        ):  # Be aware that this totally overrides some stuff
            self.deterministic_data_list = []
            for scene in self.scenes:
                for object in self.objects:
                    valid_position_adr = "datasets/apnd-dataset/deterministic_tasks/tasks_{}_positions_in_{}.json".format(
                        object, scene
                    )
                    try:
                        with open(valid_position_adr) as f:
                            data_points = json.load(f)
                    except Exception:
                        print("Failed to load", valid_position_adr)
                        continue
                    visible_data = [
                        dict(scene=scene, index=i, datapoint=data)
                        for (i, data) in enumerate(data_points[scene])
                    ]
                    self.deterministic_data_list += visible_data

        if self.sampler_mode == "test":
            random.shuffle(self.deterministic_data_list)
            self.max_tasks = self.reset_tasks = len(self.deterministic_data_list)

    def next_task(
        self, force_advance_scene: bool = False
    ) -> Optional[AbstractPickUpDropOffTask]:
        if self.max_tasks is not None and self.max_tasks <= 0:
            return None

        if self.sampler_mode != "train" and self.length <= 0:
            return None

        source_data_point, target_data_point = self.get_source_target_indices()

        scene = source_data_point["scene_name"]

        assert source_data_point["object_id"] == target_data_point["object_id"]
        assert source_data_point["scene_name"] == target_data_point["scene_name"]

        if self.env is None:
            self.env = self._create_environment()

        self.env.reset(
            scene_name=scene, agentMode="arm", agentControllerType="mid-level"
        )

        event1, event2, event3 = initialize_arm(self.env.controller)

        source_location = source_data_point
        target_location = dict(
            position=target_data_point["object_location"],
            rotation={"x": 0, "y": 0, "z": 0},
        )

        task_info = {
            "objectId": source_location["object_id"],
            "countertop_id": source_location["countertop_id"],
            "source_location": source_location,
            "target_location": target_location,
        }

        this_controller = self.env

        event = transport_wrapper(
            this_controller,
            source_location["object_id"],
            source_location["object_location"],
        )
        agent_state = source_location["agent_pose"]

        event = this_controller.step(
            dict(
                action="TeleportFull",
                standing=True,
                x=agent_state["position"]["x"],
                y=agent_state["position"]["y"],
                z=agent_state["position"]["z"],
                rotation=dict(
                    x=agent_state["rotation"]["x"],
                    y=agent_state["rotation"]["y"],
                    z=agent_state["rotation"]["z"],
                ),
                horizon=agent_state["cameraHorizon"],
            )
        )

        should_visualize_goal_start = [
            x for x in self.visualizers if issubclass(type(x), ImageVisualizer)
        ]
        if len(should_visualize_goal_start) > 0:
            task_info["visualization_source"] = source_data_point
            task_info["visualization_target"] = target_data_point

        self._last_sampled_task = self._TASK_TYPE(
            env=self.env,
            sensors=self.sensors,
            task_info=task_info,
            max_steps=self.max_steps,
            action_space=self._action_space,
            visualizers=self.visualizers,
            reward_configs=self.rewards_config,
        )

        return self._last_sampled_task

    @property
    def total_unique(self) -> Optional[Union[int, float]]:
        if self.sampler_mode == "train":
            return None
        else:
            return min(self.max_tasks, len(self.deterministic_data_list))

    @property
    def length(self) -> Union[int, float]:
        """Length.

        # Returns

        Number of total tasks remaining that can be sampled. Can be float('inf').
        """
        return (
            self.total_unique - self.sampler_index
            if self.sampler_mode != "train"
            else (float("inf") if self.max_tasks is None else self.max_tasks)
        )

    def get_source_target_indices(self):
        if self.sampler_mode == "train":
            valid_countertops = [
                k for (k, v) in self.countertop_object_to_data_id.items() if len(v) > 1
            ]
            countertop_id = random.choice(valid_countertops)
            indices = random.sample(self.countertop_object_to_data_id[countertop_id], 2)
            result = (
                self.all_possible_points[indices[0]],
                self.all_possible_points[indices[1]],
            )
        else:
            result = self.deterministic_data_list[self.sampler_index]["datapoint"]
            # ForkedPdb().set_trace()
            self.sampler_index += 1

        return result

    def calc_possible_trajectories(self, all_possible_points):

        object_to_data_id = {}

        for i in range(len(all_possible_points)):
            object_id = all_possible_points[i]["object_id"]
            object_to_data_id.setdefault(object_id, [])
            object_to_data_id[object_id].append(i)

        return object_to_data_id


class ArmPointNavTaskSampler(SimpleArmPointNavGeneralSampler):
    _TASK_TYPE = ArmPointNavTask

    def __init__(self, **kwargs) -> None:

        super(ArmPointNavTaskSampler, self).__init__(**kwargs)
        possible_initial_locations = (
            "datasets/apnd-dataset/valid_agent_initial_locations.json"
        )
        if self.sampler_mode == "test":
            possible_initial_locations = (
                "datasets/apnd-dataset/deterministic_valid_agent_initial_locations.json"
            )
        with open(possible_initial_locations) as f:
            self.possible_agent_reachable_poses = json.load(f)

    def next_task(
        self, force_advance_scene: bool = False
    ) -> Optional[AbstractPickUpDropOffTask]:
        if self.max_tasks is not None and self.max_tasks <= 0:
            return None

        if self.sampler_mode != "train" and self.length <= 0:
            return None

        source_data_point, target_data_point = self.get_source_target_indices()

        scene = source_data_point["scene_name"]

        assert source_data_point["object_id"] == target_data_point["object_id"]
        assert source_data_point["scene_name"] == target_data_point["scene_name"]

        if self.env is None:
            self.env = self._create_environment()

        self.env.reset(
            scene_name=scene, agentMode="arm", agentControllerType="mid-level"
        )

        event1, event2, event3 = initialize_arm(self.env.controller)

        source_location = source_data_point
        target_location = dict(
            position=target_data_point["object_location"],
            rotation={"x": 0, "y": 0, "z": 0},
        )


        this_controller = self.env

        event = transport_wrapper(
            this_controller,
            source_location["object_id"],
            source_location["object_location"],
        )

        agent_state = source_location[
            "initial_agent_pose"
        ]  # THe only line different from father

        event = this_controller.step(
            dict(
                action="TeleportFull",
                standing=True,
                x=agent_state["position"]["x"],
                y=agent_state["position"]["y"],
                z=agent_state["position"]["z"],
                rotation=dict(
                    x=agent_state["rotation"]["x"],
                    y=agent_state["rotation"]["y"],
                    z=agent_state["rotation"]["z"],
                ),
                horizon=agent_state["cameraHorizon"],
            )
        )

        should_visualize_goal_start = [
            x for x in self.visualizers if issubclass(type(x), ImageVisualizer)
        ]

        initial_object_info = self.env.get_object_by_id(source_location["object_id"])
        initial_agent_location = self.env.controller.last_event.metadata["agent"]
        initial_hand_state = self.env.get_absolute_hand_state()

        task_info = {
            "objectId": source_location["object_id"],
            "countertop_id": source_location["countertop_id"],
            "source_location": source_location,
            "target_location": target_location,
            'agent_initial_state': initial_agent_location,
            'initial_object_location':initial_object_info,
            'initial_hand_state': initial_hand_state,
        }

        if len(should_visualize_goal_start) > 0:
            task_info["visualization_source"] = source_data_point
            task_info["visualization_target"] = target_data_point

        self._last_sampled_task = self._TASK_TYPE(
            env=self.env,
            sensors=self.sensors,
            task_info=task_info,
            max_steps=self.max_steps,
            action_space=self._action_space,
            visualizers=self.visualizers,
            reward_configs=self.rewards_config,
        )

        return self._last_sampled_task

    def get_source_target_indices(self):
        if self.sampler_mode == "train":
            valid_countertops = [
                k for (k, v) in self.countertop_object_to_data_id.items() if len(v) > 1
            ]
            countertop_id = random.choice(valid_countertops)
            indices = random.sample(self.countertop_object_to_data_id[countertop_id], 2)
            result = (
                self.all_possible_points[indices[0]],
                self.all_possible_points[indices[1]],
            )
            scene_name = result[0]["scene_name"]
            selected_agent_init_loc = random.choice(
                self.possible_agent_reachable_poses[scene_name]
            )
            initial_agent_pose = {
                "name": "agent",
                "position": {
                    "x": selected_agent_init_loc["x"],
                    "y": selected_agent_init_loc["y"],
                    "z": selected_agent_init_loc["z"],
                },
                "rotation": {
                    "x": -0.0,
                    "y": selected_agent_init_loc["rotation"],
                    "z": 0.0,
                },
                "cameraHorizon": selected_agent_init_loc["horizon"],
                "isStanding": True,
            }
            result[0]["initial_agent_pose"] = initial_agent_pose
        else:  # we need to fix this for test set, agent init location needs to be fixed, therefore we load a fixed valid agent init that is previously randomized
            result = self.deterministic_data_list[self.sampler_index]["datapoint"]
            scene_name = self.deterministic_data_list[self.sampler_index]["scene"]
            datapoint_original_index = self.deterministic_data_list[self.sampler_index][
                "index"
            ]
            selected_agent_init_loc = self.possible_agent_reachable_poses[scene_name][
                datapoint_original_index
            ]
            initial_agent_pose = {
                "name": "agent",
                "position": {
                    "x": selected_agent_init_loc["x"],
                    "y": selected_agent_init_loc["y"],
                    "z": selected_agent_init_loc["z"],
                },
                "rotation": {
                    "x": -0.0,
                    "y": selected_agent_init_loc["rotation"],
                    "z": 0.0,
                },
                "cameraHorizon": selected_agent_init_loc["horizon"],
                "isStanding": True,
            }
            result[0]["initial_agent_pose"] = initial_agent_pose
            self.sampler_index += 1

        return result

class EasyArmPointNavTaskSampler(ArmPointNavTaskSampler):
    _TASK_TYPE = EasyArmPointNavTask

def get_all_tuples_from_list(list):
    result = []
    for first_ind in range(len(list) - 1):
        for second_ind in range(first_ind + 1, len(list)):
            result.append([list[first_ind], list[second_ind]])
    return result
