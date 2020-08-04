from typing import Any, Tuple, Optional, Dict
import typing

import gym
import numpy as np
import quaternion  # noqa # pylint: disable=unused-import
import torchvision.transforms as transforms

from rl_ai2thor.ai2thor_sensors import ScaleBothSides
from rl_base.sensor import Sensor, RGBSensor, DepthSensor
from rl_base.task import Task
from rl_robothor.robothor_environment import RoboThorEnvironment, RoboThorCachedEnvironment
from rl_robothor.robothor_tasks import PointNavTask
from utils.misc_utils import prepare_locals_for_super


class RGBSensorRoboThor(RGBSensor[RoboThorEnvironment, Task[RoboThorEnvironment]]):
    """Sensor for RGB images in RoboTHOR.

    Returns from a running RoboThorEnvironment instance, the current RGB
    frame corresponding to the agent's egocentric view.
    """

    def frame_from_env(self, env: RoboThorEnvironment) -> np.ndarray:
        return env.current_frame.copy()


class GPSCompassSensorRoboThor(Sensor[RoboThorEnvironment, PointNavTask]):
    def __init__(self, uuid: str = "target_coordinates_ind", **kwargs: Any):
        observation_space = gym.spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=(2,),
            dtype=np.float32,
        )
        super().__init__(**prepare_locals_for_super(locals()))
        self.uuid = "target_coordinates_ind"

    def _compute_pointgoal(self, source_position, source_rotation, goal_position):
        direction_vector = goal_position - source_position
        direction_vector_agent = self.quaternion_rotate_vector(
            source_rotation.inverse(), direction_vector
        )

        rho, phi = GPSCompassSensorRoboThor.cartesian_to_polar(
            direction_vector_agent[2], -direction_vector_agent[0]
        )
        return np.array([rho, phi], dtype=np.float32)

    @staticmethod
    def quaternion_from_y_angle(angle: float) -> np.quaternion:
        r"""Creates a quaternion from rotation angle around y axis
        """
        return GPSCompassSensorRoboThor.quaternion_from_coeff(
            np.array(
                [0.0, np.sin(np.pi * angle / 360.0), 0.0, np.cos(np.pi * angle / 360.0)]
            )
        )

    @staticmethod
    def quaternion_from_coeff(coeffs: np.ndarray) -> np.quaternion:
        r"""Creates a quaternions from coeffs in [x, y, z, w] format
        """
        quat = np.quaternion(0, 0, 0, 0)
        quat.real = coeffs[3]
        quat.imag = coeffs[0:3]
        return quat

    @staticmethod
    def cartesian_to_polar(x, y):
        rho = np.sqrt(x ** 2 + y ** 2)
        phi = np.arctan2(y, x)
        return rho, phi

    @staticmethod
    def quaternion_rotate_vector(quat: np.quaternion, v: np.array) -> np.array:
        r"""Rotates a vector by a quaternion
        Args:
            quaternion: The quaternion to rotate by
            v: The vector to rotate
        Returns:
            np.array: The rotated vector
        """
        vq = np.quaternion(0, 0, 0, 0)
        vq.imag = v
        return (quat * vq * quat.inverse()).imag

    def get_observation(
        self,
        env: RoboThorEnvironment,
        task: Optional[PointNavTask],
        *args: Any,
        **kwargs: Any
    ) -> Any:

        agent_state = env.agent_state()
        agent_position = np.array([agent_state[k] for k in ["x", "y", "z"]])
        rotation_world_agent = self.quaternion_from_y_angle(
            agent_state["rotation"]["y"]
        )

        goal_position = np.array([task.task_info["target"][k] for k in ["x", "y", "z"]])

        return self._compute_pointgoal(
            agent_position, rotation_world_agent, goal_position
        )


def quaternion_from_y_angle(angle: float) -> np.quaternion:
    r"""Creates a quaternion from rotation angle around y axis
    """
    return quaternion_from_coeff(np.array([0.0, np.sin(np.pi*angle/360.0), 0.0, np.cos(np.pi*angle/360.0)]))


def quaternion_from_coeff(coeffs: np.ndarray) -> np.quaternion:
    r"""Creates a quaternions from coeffs in [x, y, z, w] format
    """
    quat = np.quaternion(0, 0, 0, 0)
    quat.real = coeffs[3]
    quat.imag = coeffs[0:3]
    return quat


def cartesian_to_polar(x, y):
    rho = np.sqrt(x ** 2 + y ** 2)
    phi = np.arctan2(y, x)
    return rho, phi


def quaternion_rotate_vector(quat: np.quaternion, v: np.array) -> np.array:
    r"""Rotates a vector by a quaternion
    Args:
        quaternion: The quaternion to rotate by
        v: The vector to rotate
    Returns:
        np.array: The rotated vector
    """
    vq = np.quaternion(0, 0, 0, 0)
    vq.imag = v
    return (quat * vq * quat.inverse()).imag


class DepthSensorRoboThor(Sensor[RoboThorEnvironment, PointNavTask]):
    def __init__(self, config: Dict[str, Any], *args: Any, **kwargs: Any):
        self.config = config
        def f(x, k, default):
            return x[k] if k in x else default

        self.height: Optional[int] = f(config, "height", None)
        self.width: Optional[int] = f(config, "width", None)
        self.should_normalize = f(config, "use_resnet_normalization", False)

        assert (self.width is None) == (self.height is None), (
            "In RGBSensorThor's config, "
            "either both height/width must be None or neither."
        )

        self.norm_means = np.array([0.5], dtype=np.float32)
        self.norm_sds = np.array([[0.25]], dtype=np.float32)

        shape = None if self.height is None else (self.height, self.width, 3)
        if not self.should_normalize:
            low = 0.0
            high = 1.0
            self.observation_space = gym.spaces.Box(low=low, high=high, shape=shape)
        else:
            low = np.tile(-self.norm_means / self.norm_sds, shape[:-1] + (1,))
            high = np.tile((1 - self.norm_means) / self.norm_sds, shape[:-1] + (1,))
            self.observation_space = gym.spaces.Box(low=low, high=high)

        self.scaler = (
            None
            if self.width is None
            else ScaleBothSides(width=self.width, height=self.height)
        )

        self.to_pil = transforms.ToPILImage()
        super().__init__(config, self.observation_space, *args, **kwargs)
        self.uuid = f(config, "uuid", None)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return "depth_lowres"

    def _get_observation_space(self) -> gym.spaces.Box:
        return typing.cast(gym.spaces.Box, self.observation_space)

    def get_observation(
            self,
            env: RoboThorEnvironment,
            task: Optional[PointNavTask]=None,
            *args: Any,
            **kwargs: Any
    ) -> Any:
        depth = env.current_depth.copy()

        assert depth.dtype in [np.uint8, np.float32]

        if depth.dtype == np.uint8:
            depth = depth.astype(np.float32) / 255.0

        if self.should_normalize:
            depth -= self.norm_means
            depth /= self.norm_sds

        if self.scaler is not None and depth.shape[:2] != (self.height, self.width):
            depth = np.array(self.scaler(self.to_pil(depth)), dtype=np.float32)


class ResNetRGBSensorHabitatCache(Sensor[RoboThorCachedEnvironment, Task[RoboThorEnvironment]]):
    def __init__(self, config: Dict[str, Any], *args: Any, **kwargs: Any):
        super().__init__(config, *args, **kwargs)

        def f(x, k, default):
            return x[k] if k in x else default

        self.uuid = f(config, "uuid", None)
        self.height: Optional[int] = f(config, "height", None)
        self.width: Optional[int] = f(config, "width", None)
        self.should_normalize = f(config, "use_resnet_normalization", False)

        shape = (512, 7, 7)
        low = 0.0
        high = 1.0
        self.observation_space = gym.spaces.Box(low=low, high=high, shape=shape)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return "rgb_resnet"

    def _get_observation_space(self) -> gym.spaces.Box:
        return typing.cast(gym.spaces.Box, self.observation_space)

    def get_observation(
            self,
            env: RoboThorEnvironment,
            task: Optional[PointNavTask]=None,
            *args: Any,
            **kwargs: Any
    ) -> Any:
        return env.current_frame
