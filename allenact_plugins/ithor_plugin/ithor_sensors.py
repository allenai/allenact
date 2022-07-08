import copy
from typing import Any, Dict, Optional, Union, Sequence

import ai2thor.controller
import gym
import gym.spaces
import numpy as np
import torch

from allenact.base_abstractions.sensor import Sensor
from allenact.base_abstractions.task import Task
from allenact.embodiedai.mapping.mapping_utils.map_builders import (
    BinnedPointCloudMapBuilder,
    SemanticMapBuilder,
    ObjectHull2d,
)
from allenact.embodiedai.sensors.vision_sensors import RGBSensor
from allenact.utils.misc_utils import prepare_locals_for_super
from allenact_plugins.ithor_plugin.ithor_environment import IThorEnvironment
from allenact_plugins.ithor_plugin.ithor_tasks import ObjectNaviThorGridTask
from allenact_plugins.ithor_plugin.ithor_util import include_object_data
from allenact_plugins.robothor_plugin.robothor_environment import RoboThorEnvironment
from allenact_plugins.robothor_plugin.robothor_tasks import PointNavTask, ObjectNavTask

THOR_ENV_TYPE = Union[
    ai2thor.controller.Controller, IThorEnvironment, RoboThorEnvironment
]
THOR_TASK_TYPE = Union[
    Task[ai2thor.controller.Controller],
    Task[IThorEnvironment],
    Task[RoboThorEnvironment],
]


class RGBSensorThor(
    RGBSensor[THOR_ENV_TYPE, THOR_TASK_TYPE]
):
    """Sensor for RGB images in THOR.

    Returns from a running IThorEnvironment instance, the current RGB
    frame corresponding to the agent's egocentric view.
    """

    def frame_from_env(
        self, env: THOR_ENV_TYPE, task: Optional[THOR_TASK_TYPE],
    ) -> np.ndarray:  # type:ignore
        if isinstance(env, ai2thor.controller.Controller):
            return env.last_event.frame.copy()
        else:
            return env.current_frame.copy()


class GoalObjectTypeThorSensor(Sensor):
    def __init__(
        self,
        object_types: Sequence[str],
        target_to_detector_map: Optional[Dict[str, str]] = None,
        detector_types: Optional[Sequence[str]] = None,
        uuid: str = "goal_object_type_ind",
        **kwargs: Any,
    ):
        self.ordered_object_types = list(object_types)
        assert self.ordered_object_types == sorted(
            self.ordered_object_types
        ), "object types input to goal object type sensor must be ordered"

        self.target_to_detector_map = target_to_detector_map

        if target_to_detector_map is None:
            self.object_type_to_ind = {
                ot: i for i, ot in enumerate(self.ordered_object_types)
            }
        else:
            assert (
                detector_types is not None
            ), "Missing detector_types for map {}".format(target_to_detector_map)
            self.target_to_detector = target_to_detector_map
            self.detector_types = detector_types

            detector_index = {ot: i for i, ot in enumerate(self.detector_types)}
            self.object_type_to_ind = {
                ot: detector_index[self.target_to_detector[ot]]
                for ot in self.ordered_object_types
            }

        observation_space = self._get_observation_space()

        super().__init__(**prepare_locals_for_super(locals()))

    def _get_observation_space(self):
        if self.target_to_detector_map is None:
            return gym.spaces.Discrete(len(self.ordered_object_types))
        else:
            return gym.spaces.Discrete(len(self.detector_types))

    def get_observation(
        self,
        env: IThorEnvironment,
        task: Optional[ObjectNaviThorGridTask],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        return self.object_type_to_ind[task.task_info["object_type"]]


class TakeEndActionThorNavSensor(
    Sensor[
        Union[RoboThorEnvironment, IThorEnvironment],
        Union[ObjectNaviThorGridTask, ObjectNavTask, PointNavTask],
    ]
):
    def __init__(self, nactions: int, uuid: str, **kwargs: Any) -> None:
        self.nactions = nactions

        observation_space = self._get_observation_space()

        super().__init__(**prepare_locals_for_super(locals()))

    def _get_observation_space(self) -> gym.spaces.Discrete:
        """The observation space.

        Equals `gym.spaces.Discrete(2)` where a 0 indicates that the agent
        **should not** take the `End` action and a 1 indicates that the agent
        **should** take the end action.
        """
        return gym.spaces.Discrete(2)

    def get_observation(  # type:ignore
        self,
        env: IThorEnvironment,
        task: Union[ObjectNaviThorGridTask, ObjectNavTask, PointNavTask],
        *args,
        **kwargs,
    ) -> np.ndarray:
        if isinstance(task, ObjectNaviThorGridTask):
            should_end = task.is_goal_object_visible()
        elif isinstance(task, ObjectNavTask):
            should_end = task._is_goal_in_range()
        elif isinstance(task, PointNavTask):
            should_end = task._is_goal_in_range()
        else:
            raise NotImplementedError

        if should_end is None:
            should_end = False
        return np.array([1 * should_end], dtype=np.int64)


class RelativePositionChangeTHORSensor(
    Sensor[RoboThorEnvironment, Task[RoboThorEnvironment]]
):
    def __init__(self, uuid: str = "rel_position_change", **kwargs: Any):
        observation_space = gym.spaces.Dict(
            {
                "last_allocentric_position": gym.spaces.Box(
                    low=np.array([-np.inf, -np.inf, 0], dtype=np.float32),
                    high=np.array([np.inf, np.inf, 360], dtype=np.float32),
                    shape=(3,),
                    dtype=np.float32,
                ),
                "dx_dz_dr": gym.spaces.Box(
                    low=np.array([-np.inf, -np.inf, -360], dtype=np.float32),
                    high=np.array([-np.inf, -np.inf, 360], dtype=np.float32),
                    shape=(3,),
                    dtype=np.float32,
                ),
            }
        )
        super().__init__(**prepare_locals_for_super(locals()))

        self.last_xzr: Optional[np.ndarray] = None

    @staticmethod
    def get_relative_position_change(from_xzr: np.ndarray, to_xzr: np.ndarray):
        dx_dz_dr = to_xzr - from_xzr

        # Transform dx, dz (in global coordinates) into the relative coordinates
        # given by rotation r0=from_xzr[-2]. This requires rotating everything so that
        # r0 is facing in the positive z direction. Since thor rotations are negative
        # the usual rotation direction this means we want to rotate by r0 degrees.
        theta = np.pi * from_xzr[-1] / 180
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        dx_dz_dr = (
            np.array(
                [
                    [cos_theta, -sin_theta, 0],
                    [sin_theta, cos_theta, 0],
                    [0, 0, 1],  # Don't change dr
                ]
            )
            @ dx_dz_dr.reshape(-1, 1)
        ).reshape(-1)

        dx_dz_dr[-1] = dx_dz_dr[-1] % 360
        return dx_dz_dr

    def get_observation(
        self,
        env: RoboThorEnvironment,
        task: Optional[Task[RoboThorEnvironment]],
        *args: Any,
        **kwargs: Any,
    ) -> Any:

        if task.num_steps_taken() == 0:
            p = env.controller.last_event.metadata["agent"]["position"]
            r = env.controller.last_event.metadata["agent"]["rotation"]["y"]
            self.last_xzr = np.array([p["x"], p["z"], r % 360])

        p = env.controller.last_event.metadata["agent"]["position"]
        r = env.controller.last_event.metadata["agent"]["rotation"]["y"]
        current_xzr = np.array([p["x"], p["z"], r % 360])

        dx_dz_dr = self.get_relative_position_change(
            from_xzr=self.last_xzr, to_xzr=current_xzr
        )

        to_return = {"last_allocentric_position": self.last_xzr, "dx_dz_dr": dx_dz_dr}

        self.last_xzr = current_xzr

        return to_return


class ReachableBoundsTHORSensor(Sensor[RoboThorEnvironment, Task[RoboThorEnvironment]]):
    def __init__(self, margin: float, uuid: str = "scene_bounds", **kwargs: Any):
        observation_space = gym.spaces.Dict(
            {
                "x_range": gym.spaces.Box(
                    low=np.array([-np.inf, -np.inf], dtype=np.float32),
                    high=np.array([np.inf, np.inf], dtype=np.float32),
                    shape=(2,),
                    dtype=np.float32,
                ),
                "z_range": gym.spaces.Box(
                    low=np.array([-np.inf, -np.inf], dtype=np.float32),
                    high=np.array([np.inf, np.inf], dtype=np.float32),
                    shape=(2,),
                    dtype=np.float32,
                ),
            }
        )
        super().__init__(**prepare_locals_for_super(locals()))

        self.margin = margin
        self._bounds_cache = {}

    @staticmethod
    def get_bounds(
        controller: ai2thor.controller.Controller, margin: float,
    ) -> Dict[str, np.ndarray]:
        positions = controller.step("GetReachablePositions").metadata["actionReturn"]
        min_x = min(p["x"] for p in positions)
        max_x = max(p["x"] for p in positions)
        min_z = min(p["z"] for p in positions)
        max_z = max(p["z"] for p in positions)

        return {
            "x_range": np.array([min_x - margin, max_x + margin]),
            "z_range": np.array([min_z - margin, max_z + margin]),
        }

    def get_observation(
        self,
        env: RoboThorEnvironment,
        task: Optional[Task[RoboThorEnvironment]],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        scene_name = env.controller.last_event.metadata["sceneName"]
        if scene_name not in self._bounds_cache:
            self._bounds_cache[scene_name] = self.get_bounds(
                controller=env.controller, margin=self.margin
            )

        return copy.deepcopy(self._bounds_cache[scene_name])


class SceneBoundsTHORSensor(Sensor[RoboThorEnvironment, Task[RoboThorEnvironment]]):
    def __init__(self, uuid: str = "scene_bounds", **kwargs: Any):
        observation_space = gym.spaces.Dict(
            {
                "x_range": gym.spaces.Box(
                    low=np.array([-np.inf, -np.inf]),
                    high=np.array([np.inf, np.inf]),
                    shape=(2,),
                    dtype=np.float32,
                ),
                "z_range": gym.spaces.Box(
                    low=np.array([-np.inf, -np.inf]),
                    high=np.array([np.inf, np.inf]),
                    shape=(2,),
                    dtype=np.float32,
                ),
            }
        )
        super().__init__(**prepare_locals_for_super(locals()))

    def get_observation(
        self,
        env: RoboThorEnvironment,
        task: Optional[Task[RoboThorEnvironment]],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        scene_bounds = env.controller.last_event.metadata["sceneBounds"]
        center = scene_bounds["center"]
        size = scene_bounds["size"]

        return {
            "x_range": np.array(
                [center["x"] - size["x"] / 2, center["x"] + size["x"] / 2]
            ),
            "z_range": np.array(
                [center["z"] - size["z"] / 2, center["z"] + size["z"] / 2]
            ),
        }


class BinnedPointCloudMapTHORSensor(
    Sensor[RoboThorEnvironment, Task[RoboThorEnvironment]]
):
    observation_space = gym.spaces.Dict

    def __init__(
        self,
        fov: float,
        vision_range_in_cm: int,
        map_size_in_cm: int,
        resolution_in_cm: int,
        map_range_sensor: Sensor,
        height_bins: Sequence[float] = (0.02, 2),
        ego_only: bool = True,
        uuid: str = "binned_pc_map",
        **kwargs: Any,
    ):
        self.fov = fov
        self.vision_range_in_cm = vision_range_in_cm
        self.map_size_in_cm = map_size_in_cm
        self.resolution_in_cm = resolution_in_cm
        self.height_bins = height_bins
        self.ego_only = ego_only

        self.binned_pc_map_builder = BinnedPointCloudMapBuilder(
            fov=fov,
            vision_range_in_cm=vision_range_in_cm,
            map_size_in_cm=map_size_in_cm,
            resolution_in_cm=resolution_in_cm,
            height_bins=height_bins,
        )

        map_space = gym.spaces.Box(
            low=0,
            high=np.inf,
            shape=self.binned_pc_map_builder.binned_point_cloud_map.shape,
            dtype=np.float32,
        )

        space_dict = {
            "egocentric_update": map_space,
        }
        if not ego_only:
            space_dict["allocentric_update"] = copy.deepcopy(map_space)
            space_dict["map"] = copy.deepcopy(map_space)

        observation_space = gym.spaces.Dict(space_dict)
        super().__init__(**prepare_locals_for_super(locals()))

        self.map_range_sensor = map_range_sensor

    @property
    def device(self):
        return self.binned_pc_map_builder.device

    @device.setter
    def device(self, val: torch.device):
        self.binned_pc_map_builder.device = torch.device(val)

    def get_observation(
        self,
        env: RoboThorEnvironment,
        task: Optional[Task[RoboThorEnvironment]],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        e = env.controller.last_event
        metadata = e.metadata

        if task.num_steps_taken() == 0:
            xz_ranges_dict = self.map_range_sensor.get_observation(env=env, task=task)
            self.binned_pc_map_builder.reset(
                min_xyz=np.array(
                    [
                        xz_ranges_dict["x_range"][0],
                        0,  # TODO: Should y be different per scene?
                        xz_ranges_dict["z_range"][0],
                    ]
                )
            )

        map_dict = self.binned_pc_map_builder.update(
            depth_frame=e.depth_frame,
            camera_xyz=np.array(
                [metadata["cameraPosition"][k] for k in ["x", "y", "z"]]
            ),
            camera_rotation=metadata["agent"]["rotation"]["y"],
            camera_horizon=metadata["agent"]["cameraHorizon"],
        )
        return {k: map_dict[k] for k in self.observation_space.spaces.keys()}


class SemanticMapTHORSensor(Sensor[RoboThorEnvironment, Task[RoboThorEnvironment]]):
    observation_space = gym.spaces.Dict

    def __init__(
        self,
        fov: float,
        vision_range_in_cm: int,
        map_size_in_cm: int,
        resolution_in_cm: int,
        ordered_object_types: Sequence[str],
        map_range_sensor: Sensor,
        ego_only: bool = True,
        uuid: str = "semantic_map",
        device: torch.device = torch.device("cpu"),
        **kwargs: Any,
    ):
        self.fov = fov
        self.vision_range_in_cm = vision_range_in_cm
        self.map_size_in_cm = map_size_in_cm
        self.resolution_in_cm = resolution_in_cm
        self.ordered_object_types = ordered_object_types
        self.map_range_sensor = map_range_sensor
        self.ego_only = ego_only

        self.semantic_map_builder = SemanticMapBuilder(
            fov=fov,
            vision_range_in_cm=vision_range_in_cm,
            map_size_in_cm=map_size_in_cm,
            resolution_in_cm=resolution_in_cm,
            ordered_object_types=ordered_object_types,
            device=device,
        )

        def get_map_space(nchannels: int, size: int):
            return gym.spaces.Box(
                low=0, high=1, shape=(size, size, nchannels), dtype=np.bool,
            )

        n = len(self.ordered_object_types)
        small = self.vision_range_in_cm // self.resolution_in_cm
        big = self.semantic_map_builder.ground_truth_semantic_map.shape[0]

        space_dict = {
            "egocentric_update": get_map_space(nchannels=n, size=small,),
            "egocentric_mask": get_map_space(nchannels=1, size=small,),
        }
        if not ego_only:
            space_dict["explored_mask"] = get_map_space(nchannels=1, size=big,)
            space_dict["map"] = get_map_space(nchannels=n, size=big,)

        observation_space = gym.spaces.Dict(space_dict)
        super().__init__(**prepare_locals_for_super(locals()))

    @property
    def device(self):
        return self.semantic_map_builder.device

    @device.setter
    def device(self, val: torch.device):
        self.semantic_map_builder.device = torch.device(val)

    def get_observation(
        self,
        env: RoboThorEnvironment,
        task: Optional[Task[RoboThorEnvironment]],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        with include_object_data(env.controller):
            last_event = env.controller.last_event
            metadata = last_event.metadata

            if task.num_steps_taken() == 0:
                env.controller.step(
                    "Get2DSemanticHulls", objectTypes=self.ordered_object_types
                )
                assert env.last_event.metadata[
                    "lastActionSuccess"
                ], f"Get2DSemanticHulls failed with error '{env.last_event.metadata['lastActionSuccess']}'"

                object_id_to_hull = env.controller.last_event.metadata["actionReturn"]

                xz_ranges_dict = self.map_range_sensor.get_observation(
                    env=env, task=task
                )

                self.semantic_map_builder.reset(
                    min_xyz=np.array(
                        [
                            xz_ranges_dict["x_range"][0],
                            0,  # TODO: Should y be different per scene?
                            xz_ranges_dict["z_range"][0],
                        ]
                    ),
                    object_hulls=[
                        ObjectHull2d(
                            object_id=o["objectId"],
                            object_type=o["objectType"],
                            hull_points=object_id_to_hull[o["objectId"]],
                        )
                        for o in metadata["objects"]
                        if o["objectId"] in object_id_to_hull
                    ],
                )

            map_dict = self.semantic_map_builder.update(
                depth_frame=last_event.depth_frame,
                camera_xyz=np.array(
                    [metadata["cameraPosition"][k] for k in ["x", "y", "z"]]
                ),
                camera_rotation=metadata["agent"]["rotation"]["y"],
                camera_horizon=metadata["agent"]["cameraHorizon"],
            )
            return {
                k: map_dict[k] > 0.001 if map_dict[k].dtype != np.bool else map_dict[k]
                for k in self.observation_space.spaces.keys()
            }
