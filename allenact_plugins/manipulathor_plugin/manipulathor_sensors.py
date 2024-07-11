"""Utility classes and functions for sensory inputs used by the models."""
from typing import Any, Union, Optional

import gym
import numpy as np
from allenact.base_abstractions.sensor import Sensor
from allenact.embodiedai.sensors.vision_sensors import DepthSensor, RGBSensor
from allenact.base_abstractions.task import Task
from allenact.utils.misc_utils import prepare_locals_for_super

from allenact_plugins.manipulathor_plugin.arm_calculation_utils import (
    world_coords_to_agent_coords,
    state_dict_to_tensor,
    diff_position,
    coord_system_transform,
)
from allenact_plugins.manipulathor_plugin.manipulathor_environment import (
    ManipulaTHOREnvironment,
)


class DepthSensorThor(
    DepthSensor[
        Union[ManipulaTHOREnvironment],
        Union[Task[ManipulaTHOREnvironment]],
    ]
):
    """Sensor for Depth images in THOR.

    Returns from a running ManipulaTHOREnvironment instance, the current
    RGB frame corresponding to the agent's egocentric view.
    """

    def frame_from_env(
        self, env: ManipulaTHOREnvironment, task: Optional[Task]
    ) -> np.ndarray:
        return env.controller.last_event.depth_frame.copy()


class NoVisionSensorThor(
    RGBSensor[
        Union[ManipulaTHOREnvironment],
        Union[Task[ManipulaTHOREnvironment]],
    ]
):
    """Sensor for RGB images in THOR.

    Returns from a running ManipulaTHOREnvironment instance, the current
    RGB frame corresponding to the agent's egocentric view.
    """

    def frame_from_env(
        self, env: ManipulaTHOREnvironment, task: Optional[Task]
    ) -> np.ndarray:
        return np.zeros_like(env.current_frame)


class AgentRelativeCurrentObjectStateThorSensor(Sensor):
    def __init__(self, uuid: str = "relative_current_obj_state", **kwargs: Any):
        observation_space = gym.spaces.Box(
            low=-100, high=100, shape=(6,), dtype=np.float32
        )  # (low=-1.0, high=2.0, shape=(3, 4), dtype=np.float32)
        super().__init__(**prepare_locals_for_super(locals()))

    def get_observation(
        self, env: ManipulaTHOREnvironment, task: Task, *args: Any, **kwargs: Any
    ) -> Any:
        object_id = task.task_info["objectId"]
        current_object_state = env.get_object_by_id(object_id)
        relative_current_obj = world_coords_to_agent_coords(
            current_object_state, env.controller.last_event.metadata["agent"]
        )
        result = state_dict_to_tensor(
            dict(
                position=relative_current_obj["position"],
                rotation=relative_current_obj["rotation"],
            )
        )
        return result


class RelativeObjectToGoalSensor(Sensor):
    def __init__(
        self,
        uuid: str = "relative_obj_to_goal",
        coord_system: str = "xyz_unsigned",
        **kwargs: Any
    ):
        assert coord_system in [
            "xyz_unsigned",
            "xyz_signed",
            "polar_radian",
            "polar_trigo",
        ]
        self.coord_system = coord_system
        if coord_system == "polar_trigo":
            obs_dim = 5
        else:
            obs_dim = 3
        observation_space = gym.spaces.Box(
            low=-100, high=100, shape=(obs_dim,), dtype=np.float32
        )
        super().__init__(**prepare_locals_for_super(locals()))

    def get_observation(
        self, env: ManipulaTHOREnvironment, task: Task, *args: Any, **kwargs: Any
    ) -> Any:
        goal_obj_id = task.task_info["objectId"]
        object_info = env.get_object_by_id(goal_obj_id)
        target_state = task.task_info["target_location"]

        agent_state = env.controller.last_event.metadata["agent"]

        relative_current_obj = world_coords_to_agent_coords(object_info, agent_state)
        relative_goal_state = world_coords_to_agent_coords(target_state, agent_state)
        relative_distance = diff_position(
            relative_current_obj,
            relative_goal_state,
            absolute=False,
        )

        result = coord_system_transform(relative_distance, self.coord_system)
        return result


class InitialObjectToGoalSensor(Sensor):
    def __init__(self, uuid: str = "initial_obj_to_goal", **kwargs: Any):
        # observation_space = gym.spaces.Discrete(len(self.detector_types))
        observation_space = gym.spaces.Box(
            low=-100, high=100, shape=(3,), dtype=np.float32
        )  # (low=-1.0, high=2.0, shape=(3, 4), dtype=np.float32)
        super().__init__(**prepare_locals_for_super(locals()))

    def get_observation(
        self, env: ManipulaTHOREnvironment, task: Task, *args: Any, **kwargs: Any
    ) -> Any:
        object_source_location = task.task_info["initial_object_location"]
        target_state = task.task_info["target_location"]
        agent_state = task.task_info["agent_initial_state"]

        relative_current_obj = world_coords_to_agent_coords(
            object_source_location, agent_state
        )
        relative_goal_state = world_coords_to_agent_coords(target_state, agent_state)
        relative_distance = diff_position(relative_current_obj, relative_goal_state)
        result = state_dict_to_tensor(dict(position=relative_distance))
        return result


class DistanceObjectToGoalSensor(Sensor):
    def __init__(self, uuid: str = "distance_obj_to_goal", **kwargs: Any):
        # observation_space = gym.spaces.Discrete(len(self.detector_types))
        observation_space = gym.spaces.Box(
            low=-100, high=100, shape=(3,), dtype=np.float32
        )  # (low=-1.0, high=2.0, shape=(3, 4), dtype=np.float32)
        super().__init__(**prepare_locals_for_super(locals()))

    def get_observation(
        self, env: ManipulaTHOREnvironment, task: Task, *args: Any, **kwargs: Any
    ) -> Any:
        goal_obj_id = task.task_info["objectId"]
        object_info = env.get_object_by_id(goal_obj_id)
        target_state = task.task_info["target_location"]

        agent_state = env.controller.last_event.metadata["agent"]

        relative_current_obj = world_coords_to_agent_coords(object_info, agent_state)
        relative_goal_state = world_coords_to_agent_coords(target_state, agent_state)
        relative_distance = diff_position(relative_current_obj, relative_goal_state)
        result = state_dict_to_tensor(dict(position=relative_distance))

        result = ((result**2).sum() ** 0.5).view(1)
        return result


class RelativeAgentArmToObjectSensor(Sensor):
    def __init__(
        self,
        uuid: str = "relative_agent_arm_to_obj",
        coord_system: str = "xyz_unsigned",
        **kwargs: Any
    ):
        assert coord_system in [
            "xyz_unsigned",
            "xyz_signed",
            "polar_radian",
            "polar_trigo",
        ]
        self.coord_system = coord_system
        if coord_system == "polar_trigo":
            obs_dim = 5
        else:
            obs_dim = 3
        observation_space = gym.spaces.Box(
            low=-100, high=100, shape=(obs_dim,), dtype=np.float32
        )
        super().__init__(**prepare_locals_for_super(locals()))

    def get_observation(
        self, env: ManipulaTHOREnvironment, task: Task, *args: Any, **kwargs: Any
    ) -> Any:
        goal_obj_id = task.task_info["objectId"]
        object_info = env.get_object_by_id(goal_obj_id)
        hand_state = env.get_absolute_hand_state()

        relative_goal_obj = world_coords_to_agent_coords(
            object_info, env.controller.last_event.metadata["agent"]
        )
        relative_hand_state = world_coords_to_agent_coords(
            hand_state, env.controller.last_event.metadata["agent"]
        )
        relative_distance = diff_position(
            relative_goal_obj,
            relative_hand_state,
            absolute=False,
        )
        result = coord_system_transform(relative_distance, self.coord_system)
        return result


class InitialAgentArmToObjectSensor(Sensor):
    def __init__(self, uuid: str = "initial_agent_arm_to_obj", **kwargs: Any):
        observation_space = gym.spaces.Box(
            low=-100, high=100, shape=(3,), dtype=np.float32
        )  # (low=-1.0, high=2.0, shape=(3, 4), dtype=np.float32)
        super().__init__(**prepare_locals_for_super(locals()))

    def get_observation(
        self, env: ManipulaTHOREnvironment, task: Task, *args: Any, **kwargs: Any
    ) -> Any:
        object_source_location = task.task_info["initial_object_location"]
        initial_hand_state = task.task_info["initial_hand_state"]

        relative_goal_obj = world_coords_to_agent_coords(
            object_source_location, env.controller.last_event.metadata["agent"]
        )
        relative_hand_state = world_coords_to_agent_coords(
            initial_hand_state, env.controller.last_event.metadata["agent"]
        )
        relative_distance = diff_position(relative_goal_obj, relative_hand_state)
        result = state_dict_to_tensor(dict(position=relative_distance))

        return result


class DistanceAgentArmToObjectSensor(Sensor):
    def __init__(self, uuid: str = "distance_agent_arm_to_obj", **kwargs: Any):
        observation_space = gym.spaces.Box(
            low=-100, high=100, shape=(3,), dtype=np.float32
        )  # (low=-1.0, high=2.0, shape=(3, 4), dtype=np.float32)
        super().__init__(**prepare_locals_for_super(locals()))

    def get_observation(
        self, env: ManipulaTHOREnvironment, task: Task, *args: Any, **kwargs: Any
    ) -> Any:
        goal_obj_id = task.task_info["objectId"]
        object_info = env.get_object_by_id(goal_obj_id)
        hand_state = env.get_absolute_hand_state()

        relative_goal_obj = world_coords_to_agent_coords(
            object_info, env.controller.last_event.metadata["agent"]
        )
        relative_hand_state = world_coords_to_agent_coords(
            hand_state, env.controller.last_event.metadata["agent"]
        )
        relative_distance = diff_position(relative_goal_obj, relative_hand_state)
        result = state_dict_to_tensor(dict(position=relative_distance))

        result = ((result**2).sum() ** 0.5).view(1)
        return result


class PickedUpObjSensor(Sensor):
    def __init__(self, uuid: str = "pickedup_object", **kwargs: Any):
        observation_space = gym.spaces.Box(
            low=0, high=1, shape=(1,), dtype=np.float32
        )  # (low=-1.0, high=2.0, shape=(3, 4), dtype=np.float32)
        super().__init__(**prepare_locals_for_super(locals()))

    def get_observation(
        self, env: ManipulaTHOREnvironment, task: Task, *args: Any, **kwargs: Any
    ) -> Any:
        return task.object_picked_up
