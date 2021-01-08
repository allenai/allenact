from typing import Any, Optional
from multiprocessing.managers import BaseManager
from contextlib import contextmanager
from tempfile import mkstemp
import os
import psutil
import math
import yaml

from orbslam2 import SLAM
from scipy.spatial.transform import Rotation as R
import numpy as np
import quaternion
import gym

from core.base_abstractions.task import SubTaskType
from core.base_abstractions.sensor import Sensor, RGBSensor, DepthSensor
from core.base_abstractions.misc import EnvType
from plugins.robothor_plugin.robothor_environment import RoboThorEnvironment
from plugins.robothor_plugin.robothor_sensors import GPSCompassSensorRoboThor
from plugins.robothor_plugin.robothor_tasks import PointNavTask
from utils.misc_utils import prepare_locals_for_super


class ProcessManager(BaseManager):
    pass

ProcessManager.register('SLAM', SLAM)

class ORBSLAMSensor(Sensor[EnvType, SubTaskType]):
    def __init__(self, rgb_sensor: RGBSensor, depth_sensor: DepthSensor, vocab_file: str, use_slam_viewer: bool, uuid: str = "orbslam", **kwargs: Any):
        self.rgb_sensor = rgb_sensor
        self.depth_sensor = depth_sensor
        self.vocab_file = vocab_file
        self.use_slam_viewer = use_slam_viewer
        observation_space = self._get_observation_space()
        super().__init__(**prepare_locals_for_super(locals()))
        self.mem_limit = 4096
        self.manager = None
        self.slam = None

    def _get_observation_space(self):
        raise NotImplementedError()

    def _get_settings(self, env):
        raise NotImplementedError()

    def _get_env_pose(self, env):
        raise NotImplementedError()

    def _initialize_slam(self, env):
        if self.manager is not None:
            self.manager.shutdown()
        self.manager = ProcessManager()
        self.manager.start()

        settings_file = mkstemp()[1]
        with open(settings_file, 'w') as file:
            file.write('%YAML:1.0')
            file.write(yaml.dump(self._get_settings(env)))
        initial_pose = self._get_env_pose(env)

        self.slam = self.manager.SLAM(self.vocab_file, settings_file, 'rgbd', initial_pose, self.use_slam_viewer)

        os.remove(settings_file)

    def _memory_exceeded(self):
        if self.mem_limit is None:
            return False
        manager_pid = self.manager._process.pid
        mem_allocated = psutil.Process(manager_pid).memory_info().rss / (1024 ** 2)  # in MB
        return mem_allocated > self.mem_limit

    def _reset_map(self, new_pose=None):
        self.slam.reset(new_pose)

    def _stop(self):
        self.manager.shutdown()


class ORBSLAMCompassSensorRoboThor(ORBSLAMSensor[RoboThorEnvironment, PointNavTask]):

    def _get_observation_space(self):
        return gym.spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=(2,),
            dtype=np.float32,
        )

    def _get_settings(self, env):
        w, h, fov = (env.last_event.metadata[m] for m in ['screenWidth', 'screenHeight', 'fov'])
        f_x = w / (2 * math.tan(math.radians(fov / 2)))
        f_y = h / (2 * math.tan(math.radians(fov / 2)))
        c_x, c_y = w / 2, h / 2
        # check Camera.fps, Camera.bf, ThDepth, DepthMapFactor
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

    def _get_env_pose(self, env):
        agent_state = env.agent_state()
        agent_position = np.array([agent_state[k] for k in 'xyz'])
        agent_rotation = np.array([agent_state['rotation'][k] for k in 'xyz'])
        pose = np.eye(4, dtype=np.float32)
        pose[:3, 3] = agent_position
        pose[:3, :3] = R.from_euler('xyz', agent_rotation, degrees=True).as_matrix()
        return pose

    def get_observation(self, env: RoboThorEnvironment, task: Optional[PointNavTask], *args: Any, **kwargs: Any) -> Any:
        if task.num_steps_taken() == 0:
            if self.slam is None or self._memory_exceeded():
                self._initialize_slam(env)
            else:
                new_pose = self._get_env_pose(env)
                self._reset_map(new_pose)

        frame = np.uint8(self.rgb_sensor.get_observation(env, task, *args, **kwargs) * 255)
        depth = np.float32(self.depth_sensor.get_observation(env, task, *args, **kwargs)[:, :, 0])
        self.slam.track(frame, depth)

        pose = self.slam.get_world_pose()
        agent_position = pose[:3, 3]
        rotation_world_agent = np.quaternion(*R.from_matrix(pose[:3, :3]).as_quat())
        goal_position = np.array([task.task_info["target"][k] for k in 'xyz'])

        goal = GPSCompassSensorRoboThor.compute_pointgoal(
            agent_position, rotation_world_agent, goal_position
        )
        return goal
