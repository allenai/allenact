from typing import Any, Dict, Optional

import gym
import numpy as np
import quaternion  # noqa # pylint: disable=unused-import
import typing
from torchvision import transforms

from rl_robothor.robothor_environment import RoboThorEnvironment
from rl_robothor.robothor_tasks import PointNavTask
from rl_base.sensor import Sensor
from rl_ai2thor.ai2thor_sensors import ScaleBothSides


class GPSCompassSensorRoboThor(Sensor[RoboThorEnvironment, PointNavTask]):
    def __init__(self, config: Dict[str, Any], *args: Any, **kwargs: Any):
        super().__init__(config, *args, **kwargs)

        self.observation_space = gym.spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=(2,),
            dtype=np.float32,)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return "target_coordinates_ind"

    def _get_observation_space(self) -> gym.spaces.Box:
        return typing.cast(gym.spaces.Box, self.observation_space)

    def _compute_pointgoal(
            self, source_position, source_rotation, goal_position
    ):
        direction_vector = goal_position - source_position
        direction_vector_agent = quaternion_rotate_vector(
            source_rotation.inverse(), direction_vector
        )

        rho, phi = cartesian_to_polar(
            direction_vector_agent[2], -direction_vector_agent[0]
        )
        return np.array([rho, phi], dtype=np.float32)

    def get_observation(
        self,
        env: RoboThorEnvironment,
        task: Optional[PointNavTask],
        *args: Any,
        **kwargs: Any
    ) -> Any:

        agent_state = env.agent_state()
        agent_position = np.array([agent_state[k] for k in ['x', 'y', 'z']])
        rotation_world_agent = quaternion_from_y_angle(agent_state['rotation']['y'])

        goal_position = np.array([task.task_info["target"][k] for k in ['x', 'y', 'z']])

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
        super().__init__(config, *args, **kwargs)

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

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return "depth"

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

        depth = np.expand_dims(depth, 2)

        return depth
