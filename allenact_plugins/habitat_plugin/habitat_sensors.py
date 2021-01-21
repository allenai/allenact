from typing import Any, Optional, Tuple

import gym
import numpy as np
from pyquaternion import Quaternion

from allenact.base_abstractions.sensor import (
    Sensor,
    RGBSensor,
    DepthSensor,
)
from allenact.base_abstractions.task import Task
from allenact.utils.misc_utils import prepare_locals_for_super
from allenact_plugins.habitat_plugin.habitat_environment import HabitatEnvironment
from allenact_plugins.habitat_plugin.habitat_tasks import PointNavTask  # type: ignore


class RGBSensorHabitat(RGBSensor[HabitatEnvironment, Task[HabitatEnvironment]]):
    # For backwards compatibility
    def __init__(
        self,
        use_resnet_normalization: bool = False,
        mean: Optional[np.ndarray] = np.array(
            [[[0.485, 0.456, 0.406]]], dtype=np.float32
        ),
        stdev: Optional[np.ndarray] = np.array(
            [[[0.229, 0.224, 0.225]]], dtype=np.float32
        ),
        height: Optional[int] = None,
        width: Optional[int] = None,
        uuid: str = "rgb",
        output_shape: Optional[Tuple[int, ...]] = None,
        output_channels: int = 3,
        unnormalized_infimum: float = 0.0,
        unnormalized_supremum: float = 1.0,
        scale_first: bool = True,
        **kwargs: Any
    ):
        super().__init__(**prepare_locals_for_super(locals()))

    def frame_from_env(self, env: HabitatEnvironment) -> np.ndarray:
        return env.current_frame["rgb"].copy()


class DepthSensorHabitat(DepthSensor[HabitatEnvironment, Task[HabitatEnvironment]]):
    # For backwards compatibility
    def __init__(
        self,
        use_resnet_normalization: Optional[bool] = None,
        use_normalization: Optional[bool] = None,
        mean: Optional[np.ndarray] = np.array([[0.5]], dtype=np.float32),
        stdev: Optional[np.ndarray] = np.array([[0.25]], dtype=np.float32),
        height: Optional[int] = None,
        width: Optional[int] = None,
        uuid: str = "depth",
        output_shape: Optional[Tuple[int, ...]] = None,
        output_channels: int = 1,
        unnormalized_infimum: float = 0.0,
        unnormalized_supremum: float = 5.0,
        scale_first: bool = False,
        **kwargs: Any
    ):
        # Give priority to use_normalization, but use_resnet_normalization for backward compat. if not set
        if use_resnet_normalization is not None and use_normalization is None:
            use_normalization = use_resnet_normalization
        elif use_normalization is None:
            use_normalization = False

        super().__init__(**prepare_locals_for_super(locals()))

    def frame_from_env(self, env: HabitatEnvironment) -> np.ndarray:
        return env.current_frame["depth"].copy()


class TargetCoordinatesSensorHabitat(Sensor[HabitatEnvironment, PointNavTask]):
    def __init__(
        self, coordinate_dims: int, uuid: str = "target_coordinates_ind", **kwargs: Any
    ):
        self.coordinate_dims = coordinate_dims

        observation_space = self._get_observation_space()

        super().__init__(**prepare_locals_for_super(locals()))

    def _get_observation_space(self):
        # Distance is a non-negative real and angle is normalized to the range (-Pi, Pi] or [-Pi, Pi)
        return gym.spaces.Box(
            np.float32(-3.15), np.float32(1000), shape=(self.coordinate_dims,)
        )

    def get_observation(
        self,
        env: HabitatEnvironment,
        task: Optional[PointNavTask],
        *args: Any,
        **kwargs: Any
    ) -> Any:
        frame = env.current_frame
        goal = frame["pointgoal_with_gps_compass"]
        return goal


class TargetObjectSensorHabitat(Sensor[HabitatEnvironment, PointNavTask]):
    def __init__(self, uuid: str = "target_object_id", **kwargs: Any):
        observation_space = self._get_observation_space()

        super().__init__(**prepare_locals_for_super(locals()))

    def _get_observation_space(self):
        return gym.spaces.Discrete(38)

    def get_observation(
        self,
        env: HabitatEnvironment,
        task: Optional[PointNavTask],
        *args: Any,
        **kwargs: Any
    ) -> Any:
        frame = env.current_frame
        goal = frame["objectgoal"][0]
        return goal


class AgentCoordinatesSensorHabitat(Sensor[HabitatEnvironment, PointNavTask]):
    def __init__(self, uuid: str = "agent_position_and_rotation", **kwargs: Any):
        observation_space = self._get_observation_space()

        super().__init__(**prepare_locals_for_super(locals()))

    def _get_observation_space(self):
        return gym.spaces.Box(np.float32(-1000), np.float32(1000), shape=(4,))

    def get_observation(
        self,
        env: HabitatEnvironment,
        task: Optional[PointNavTask],
        *args: Any,
        **kwargs: Any
    ) -> Any:
        position = env.env.sim.get_agent_state().position
        quaternion = Quaternion(env.env.sim.get_agent_state().rotation.components)
        return np.array([position[0], position[1], position[2], quaternion.radians])
