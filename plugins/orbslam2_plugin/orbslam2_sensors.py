import os
import math
import yaml
from typing import Any, Optional
from functools import reduce
from tempfile import mkstemp

import gym
import numpy as np

import orbslam2
from core.base_abstractions.task import SubTaskType, Task
from core.base_abstractions.misc import EnvType
from core.base_abstractions.sensor import Sensor, RGBSensor, DepthSensor
from plugins.robothor_plugin.robothor_sensors import RGBSensorRoboThor, DepthSensorRoboThor, GPSCompassSensorRoboThor
from plugins.robothor_plugin.robothor_environment import RoboThorEnvironment
from utils.misc_utils import prepare_locals_for_super
from plugins.robothor_plugin.robothor_tasks import ObjectNavTask, PointNavTask


# Modified from pytorch3d.transforms
def euler_angles_to_matrix(euler_angles):
    """ Input: Euler angles in ZYX order """

    def _axis_angle_rotation(axis, angle):
        cos, sin = np.cos(angle), np.sin(angle)
        one, zero = np.ones_like(angle), np.zeros_like(angle)
        if axis == "X":
            R_flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
        elif axis == "Y":
            R_flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
        elif axis == "Z":
            R_flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)
        return np.stack(R_flat, -1).reshape(angle.shape + (3, 3))

    matrices = map(_axis_angle_rotation, 'ZYX', euler_angles)
    return reduce(np.matmul, matrices)

def matrix_to_euler_angles(matrix):
    """ Output: Euler angles in XYZ order """

    def _angle_from_tan(axis, other_axis, data, horizontal):
        i1, i2 = {"X": (2, 1), "Y": (0, 2), "Z": (1, 0)}[axis]
        if horizontal:
            i2, i1 = i1, i2
        even = (axis + other_axis) in ["XY", "YZ", "ZX"]
        if horizontal == even:
            return np.arctan2(data[..., i1], data[..., i2])
        return np.arctan2(-data[..., i2], data[..., i1])

    o = (
        _angle_from_tan('X', 'Y', matrix[..., 2], False),
        np.arcsin(matrix[..., 0, 2] * (-1.0 if 0 - 2 in [-1, 2] else 1.0)),
        _angle_from_tan('Z', 'Y', matrix[..., 0, :], True)
    )
    return np.stack(o, -1)
##

def compute_pose(agent_position, agent_rotation):
    pose = np.eye(4, dtype=np.float32)
    pose[:3, 3] = np.array([agent_position[d] for d in 'xyz'])
    pose[:3, :3] = euler_angles_to_matrix(np.deg2rad(
        np.array([agent_rotation[d] for d in 'zyx'])
    ))
    return pose

def state_to_pose(agent_state):
    agent_position = {k : agent_state[k] for k in 'xyz'}
    agent_rotation = {k : agent_state['rotation'][k] for k in 'xyz'}
    pose = compute_pose(agent_position, agent_rotation)
    return pose

# check Camera.fps, Camera.bf, ThDepth, DepthMapFactor
# parameterize ORBextractor
def init_settings(w, h, fov):
    f_x = w / (2 * math.tan(math.radians(fov / 2)))
    f_y = h / (2 * math.tan(math.radians(fov / 2)))
    c_x, c_y = w / 2, h / 2
    return {
        'Camera.fx': f_x, 'Camera.fy': f_y, 'Camera.cx': c_x, 'Camera.cy': c_y,
        'Camera.k1': 0.0, 'Camera.k2': 0.0, 'Camera.p1': 0.0, 'Camera.p2': 0.0,
        'Camera.width': w, 'Camera.height': h, 'Camera.fps': 0.0, 'Camera.bf': 40.0, 'Camera.RGB': 1,
        'ThDepth': 200.0, 'DepthMapFactor': 1.0,
        'ORBextractor.nFeatures': 1000, 'ORBextractor.scaleFactor': 1.2, 'ORBextractor.nLevels': 8,
        'ORBextractor.iniThFAST': 20, 'ORBextractor.minThFAST': 1,
        'Viewer.KeyFrameSize': 0.05, 'Viewer.KeyFrameLineWidth': 1, 'Viewer.GraphLineWidth': 0.9,
        'Viewer.PointSize':2, 'Viewer.CameraSize': 0.08, 'Viewer.CameraLineWidth': 3,
        'Viewer.ViewpointX': 0, 'Viewer.ViewpointY': -0.7, 'Viewer.ViewpointZ': -1.8, 'Viewer.ViewpointF': 500
    }

class ORBSLAMSensor(Sensor[EnvType, SubTaskType]):
    def __init__(self, rgb_sensor: RGBSensor, depth_sensor: DepthSensor, vocab_file: str, use_slam_viewer: bool, uuid: str = "orbslam", **kwargs: Any):
        observation_space = self._get_observation_space()
        super().__init__(**prepare_locals_for_super(locals()))
        self.rgb_sensor, self.depth_sensor = rgb_sensor, depth_sensor
        self.vocab_file, self.use_slam_viewer = vocab_file, use_slam_viewer
        self.slam_system = None

    def _get_observation_space(self):
        raise NotImplementedError()

    def initialize_slam(self, env):
        raise NotImplementedError()

    def reset(self, agent_state):
        if self.slam_system is not None:
            pose = state_to_pose(agent_state)
            self.slam_system.reset(pose)

    def stop(self):
        self.slam_system.shutdown()


class ORBSLAMCompassSensorRoboThor(ORBSLAMSensor[RoboThorEnvironment, PointNavTask]):

    def __init__(self, rgb_sensor: RGBSensor, depth_sensor: DepthSensor, vocab_file: str, use_slam_viewer: bool, uuid: str = "target_coordinates_ind", **kwargs: Any):
        super().__init__(rgb_sensor, depth_sensor, vocab_file, use_slam_viewer, uuid, **kwargs)

    def _get_observation_space(self):
        return gym.spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=(2,),
            dtype=np.float32,
        )

    def initialize_slam(self, env):
        settings_file = mkstemp()[1]
        with open(settings_file, 'w') as file:
            file.write('%YAML:1.0')
            slam_settings = init_settings(
                *[env.last_event.metadata[m] for m in ['screenWidth', 'screenHeight', 'fov']]
            )
            file.write(yaml.dump(slam_settings))
        initial_pose = state_to_pose(env.agent_state())
        slam_system = orbslam2.SLAM(self.vocab_file, settings_file, 'rgbd', initial_pose, self.use_slam_viewer)
        os.remove(settings_file)
        return slam_system

    def get_observation(self, env: RoboThorEnvironment, task: Optional[PointNavTask], *args: Any, **kwargs: Any) -> Any:
        if self.slam_system is None:
            self.slam_system = self.initialize_slam(env)

        frame = np.uint8(self.rgb_sensor.get_observation(env, task, *args, **kwargs) * 255)
        depth = np.float32(self.depth_sensor.get_observation(env, task, *args, **kwargs)[:, :, 0])
        self.slam_system.track(frame, depth)

        pose = self.slam_system.get_world_pose()
        agent_position = pose[:3, 3]
        agent_rotation = np.rad2deg(matrix_to_euler_angles(pose[:3, :3]))
        rotation_world_agent = GPSCompassSensorRoboThor.quaternion_from_y_angle(agent_rotation[1])
        goal_position = np.array([task.task_info["target"][k] for k in 'xyz'])

        goal = GPSCompassSensorRoboThor._compute_pointgoal(GPSCompassSensorRoboThor,
            agent_position, rotation_world_agent, goal_position
        )
        return goal
