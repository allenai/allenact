import typing
from typing import Any, Dict, Optional

import gym
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from pyquaternion import Quaternion
from torchvision import transforms

from rl_base.sensor import (
    Sensor,
    RGBSensor,
    RGBResNetSensor,
    DepthSensor,
    DepthResNetSensor,
)
from rl_base.task import Task
from rl_habitat.habitat_environment import HabitatEnvironment
from rl_habitat.habitat_tasks import PointNavTask, HabitatTask  # type: ignore
from utils.tensor_utils import ScaleBothSides


class RGBSensorHabitat(RGBSensor[HabitatEnvironment, Task[HabitatEnvironment]]):
    def __init__(
        self, config: Dict[str, Any], scale_first=False, *args: Any, **kwargs: Any
    ):
        super().__init__(config, scale_first, *args, **kwargs)

    def frame_from_env(self, env: HabitatEnvironment) -> np.ndarray:
        return env.current_frame["rgb"].copy()


class RGBResNetSensorHabitat(
    RGBResNetSensor[HabitatEnvironment, Task[HabitatEnvironment]]
):
    def __init__(
        self, config: Dict[str, Any], scale_first=False, *args: Any, **kwargs: Any
    ):
        def f(x, k, default):
            return x[k] if k in x else default

        config["uuid"] = f(config, "uuid", "rgb")

        super().__init__(config, scale_first, *args, **kwargs)

    def frame_from_env(self, env: HabitatEnvironment) -> np.ndarray:
        return env.current_frame["rgb"].copy()


class DepthSensorHabitat(DepthSensor[HabitatEnvironment, Task[HabitatEnvironment]]):
    def __init__(
        self, config: Dict[str, Any], scale_first=False, *args: Any, **kwargs: Any
    ):
        def f(x, k, default):
            return x[k] if k in x else default

        # Backwards compatibility
        config["use_normalization"] = f(
            config, "use_normalization", f(config, "use_resnet_normalization", False)
        )

        super().__init__(config, scale_first, *args, **kwargs)

    def frame_from_env(self, env: HabitatEnvironment) -> np.ndarray:
        return env.current_frame["depth"].copy()


class DepthResNetSensorHabitat(
    DepthResNetSensor[HabitatEnvironment, Task[HabitatEnvironment]]
):
    def __init__(
        self, config: Dict[str, Any], scale_first=False, *args: Any, **kwargs: Any
    ):
        def f(x, k, default):
            return x[k] if k in x else default

        config["uuid"] = f(config, "uuid", "depth")

        super().__init__(config, scale_first, *args, **kwargs)

    def frame_from_env(self, env: HabitatEnvironment) -> np.ndarray:
        return env.current_frame["depth"].copy()


class TargetCoordinatesSensorHabitat(Sensor[HabitatEnvironment, PointNavTask]):
    def __init__(self, config: Dict[str, Any], *args: Any, **kwargs: Any):
        super().__init__(config, *args, **kwargs)

        # Distance is a non-negative real and angle is normalized to the range (-Pi, Pi] or [-Pi, Pi)
        self.observation_space = gym.spaces.Box(
            np.float32(-3.15), np.float32(1000), shape=(config["coordinate_dims"],)
        )

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return "target_coordinates_ind"

    def _get_observation_space(self) -> gym.spaces.Discrete:
        return typing.cast(gym.spaces.Discrete, self.observation_space)

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
    def __init__(self, config: Dict[str, Any], *args: Any, **kwargs: Any):
        super().__init__(config, *args, **kwargs)

        self.observation_space = gym.spaces.Discrete(38)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return "target_object_id"

    def _get_observation_space(self) -> gym.spaces.Discrete:
        return typing.cast(gym.spaces.Discrete, self.observation_space)

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
    def __init__(self, config: Dict[str, Any], *args: Any, **kwargs: Any):
        super().__init__(config, *args, **kwargs)

        self.observation_space = gym.spaces.Box(
            np.float32(-1000), np.float32(1000), shape=(4,)
        )

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return "agent_position_and_rotation"

    def _get_observation_space(self) -> gym.spaces.Discrete:
        return typing.cast(gym.spaces.Discrete, self.observation_space)

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
