import os
import math
import yaml
from typing import Any, Optional
from tempfile import mkstemp

import gym
import numpy as np

import orbslam2
from core.base_abstractions.task import SubTaskType, Task
from core.base_abstractions.sensor import Sensor, RGBSensor, DepthSensor
from core.base_abstractions.misc import EnvType
from plugins.robothor_plugin.robothor_environment import RoboThorEnvironment
from plugins.robothor_plugin.robothor_sensors import RGBSensorRoboThor, DepthSensorRoboThor, GPSCompassSensorRoboThor
from plugins.robothor_plugin.robothor_tasks import PointNavTask
from plugins.orbslam2_plugin.orbslam2_util import state_to_pose, matrix_to_euler_angles
from utils.misc_utils import prepare_locals_for_super


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
