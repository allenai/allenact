import copy
import glob
import math
import pickle
import random
from typing import Any, Optional, Dict, List, Union, Tuple, Collection

from ai2thor.fifo_server import FifoServer
import ai2thor.server
import numpy as np
from ai2thor.controller import Controller
from ai2thor.util import metrics

from allenact.utils.cache_utils import (
    DynamicDistanceCache,
    pos_to_str_for_cache,
    str_to_pos_for_cache,
)
from allenact.utils.experiment_utils import recursive_update
from allenact.utils.system import get_logger


class RoboThorEnvironment:
    """Wrapper for the robo2thor controller providing additional functionality
    and bookkeeping.

    See [here](https://ai2thor.allenai.org/robothor/documentation) for comprehensive
     documentation on RoboTHOR.

    # Attributes

    controller : The AI2THOR controller.
    config : The AI2THOR controller configuration
    """

    def __init__(self, all_metadata_available: bool = True, **kwargs):
        self.config = dict(
            rotateStepDegrees=30.0,
            visibilityDistance=1.0,
            gridSize=0.25,
            agentType="stochastic",
            continuousMode=True,
            snapToGrid=False,
            agentMode="bot",
            width=640,
            height=480,
            agentCount=1,
            server_class=FifoServer,
        )

        if "agentCount" in kwargs:
            assert kwargs["agentCount"] > 0

        recursive_update(self.config, {**kwargs, "agentMode": "bot"})
        self.controller = Controller(**self.config,)
        self.all_metadata_available = all_metadata_available

        self.scene_to_reachable_positions: Optional[Dict[str, Any]] = None
        self.distance_cache: Optional[DynamicDistanceCache] = None

        if self.all_metadata_available:
            self.scene_to_reachable_positions = {
                self.scene_name: copy.deepcopy(self.currently_reachable_points)
            }
            assert len(self.scene_to_reachable_positions[self.scene_name]) > 10

            self.distance_cache = DynamicDistanceCache(rounding=1)

        self.agent_count = self.config["agentCount"]

    def initialize_grid_dimensions(
        self, reachable_points: Collection[Dict[str, float]]
    ) -> Tuple[int, int, int, int]:
        """Computes bounding box for reachable points quantized with the
        current gridSize."""
        points = {
            (
                round(p["x"] / self.config["gridSize"]),
                round(p["z"] / self.config["gridSize"]),
            ): p
            for p in reachable_points
        }

        assert len(reachable_points) == len(points)

        xmin, xmax = min([p[0] for p in points]), max([p[0] for p in points])
        zmin, zmax = min([p[1] for p in points]), max([p[1] for p in points])

        return xmin, xmax, zmin, zmax

    def set_object_filter(self, object_ids: List[str]):
        self.controller.step("SetObjectFilter", objectIds=object_ids, renderImage=False)

    def reset_object_filter(self):
        self.controller.step("ResetObjectFilter", renderImage=False)

    def path_from_point_to_object_type(
        self, point: Dict[str, float], object_type: str, allowed_error: float
    ) -> Optional[List[Dict[str, float]]]:
        event = self.controller.step(
            action="GetShortestPath",
            objectType=object_type,
            position=point,
            allowedError=allowed_error,
        )
        if event.metadata["lastActionSuccess"]:
            return event.metadata["actionReturn"]["corners"]
        else:
            get_logger().debug(
                "Failed to find path for {} in {}. Start point {}, agent state {}.".format(
                    object_type,
                    self.controller.last_event.metadata["sceneName"],
                    point,
                    self.agent_state(),
                )
            )
            return None

    def distance_from_point_to_object_type(
        self, point: Dict[str, float], object_type: str, allowed_error: float
    ) -> float:
        """Minimal geodesic distance from a point to an object of the given
        type.

        It might return -1.0 for unreachable targets.
        """
        path = self.path_from_point_to_object_type(point, object_type, allowed_error)
        if path:
            # Because `allowed_error != 0` means that the path returned above might not start
            # at `point`, we explicitly add any offset there is.
            s_dist = math.sqrt(
                (point["x"] - path[0]["x"]) ** 2 + (point["z"] - path[0]["z"]) ** 2
            )
            return metrics.path_distance(path) + s_dist
        return -1.0

    def distance_to_object_type(self, object_type: str, agent_id: int = 0) -> float:
        """Minimal geodesic distance to object of given type from agent's
        current location.

        It might return -1.0 for unreachable targets.
        """
        assert 0 <= agent_id < self.agent_count
        assert self.all_metadata_available, (
            "`distance_to_object_type` cannot be called when `self.all_metadata_available` is `False`."
        )

        def retry_dist(position: Dict[str, float], object_type: str):
            allowed_error = 0.05
            debug_log = ""
            d = -1.0
            while allowed_error < 2.5:
                d = self.distance_from_point_to_object_type(
                    position, object_type, allowed_error
                )
                if d < 0:
                    debug_log = (
                        f"In scene {self.scene_name}, could not find a path from {position} to {object_type} with"
                        f" {allowed_error} error tolerance. Increasing this tolerance to"
                        f" {2 * allowed_error} any trying again."
                    )
                    allowed_error *= 2
                else:
                    break
            if d < 0:
                get_logger().warning(
                    f"In scene {self.scene_name}, could not find a path from {position} to {object_type}"
                    f" with {allowed_error} error tolerance. Returning a distance of -1."
                )
            elif debug_log != "":
                get_logger().debug(debug_log)
            return d

        return self.distance_cache.find_distance(
            self.controller.last_event.events[agent_id].metadata["agent"]["position"],
            object_type,
            retry_dist,
        )

    def path_from_point_to_point(
        self, position: Dict[str, float], target: Dict[str, float], allowedError: float
    ) -> Optional[List[Dict[str, float]]]:
        try:
            return self.controller.step(
                action="GetShortestPathToPoint",
                position=position,
                x=target["x"],
                y=target["y"],
                z=target["z"],
                allowedError=allowedError,
            ).metadata["actionReturn"]["corners"]
        except:
            get_logger().debug(
                "Failed to find path for {} in {}. Start point {}, agent state {}.".format(
                    target,
                    self.controller.last_event.metadata["sceneName"],
                    position,
                    self.agent_state(),
                )
            )
            return None

    def distance_from_point_to_point(
        self, position: Dict[str, float], target: Dict[str, float], allowed_error: float
    ) -> float:
        path = self.path_from_point_to_point(position, target, allowed_error)
        if path:
            # Because `allowed_error != 0` means that the path returned above might not start
            # or end exactly at the position/target points, we explictly add any offset there is.
            s_dist = math.sqrt(
                (position["x"] - path[0]["x"]) ** 2
                + (position["z"] - path[0]["z"]) ** 2
            )
            t_dist = math.sqrt(
                (target["x"] - path[-1]["x"]) ** 2 + (target["z"] - path[-1]["z"]) ** 2
            )
            return metrics.path_distance(path) + s_dist + t_dist
        return -1.0

    def distance_to_point(self, target: Dict[str, float], agent_id: int = 0) -> float:
        """Minimal geodesic distance to end point from agent's current
        location.

        It might return -1.0 for unreachable targets.
        """
        assert 0 <= agent_id < self.agent_count
        assert self.all_metadata_available, (
            "`distance_to_object_type` cannot be called when `self.all_metadata_available` is `False`."
        )

        def retry_dist(position: Dict[str, float], target: Dict[str, float]):
            allowed_error = 0.05
            debug_log = ""
            d = -1.0
            while allowed_error < 2.5:
                d = self.distance_from_point_to_point(position, target, allowed_error)
                if d < 0:
                    debug_log = (
                        f"In scene {self.scene_name}, could not find a path from {position} to {target} with"
                        f" {allowed_error} error tolerance. Increasing this tolerance to"
                        f" {2 * allowed_error} any trying again."
                    )
                    allowed_error *= 2
                else:
                    break
            if d < 0:
                get_logger().warning(
                    f"In scene {self.scene_name}, could not find a path from {position} to {target}"
                    f" with {allowed_error} error tolerance. Returning a distance of -1."
                )
            elif debug_log != "":
                get_logger().debug(debug_log)
            return d

        return self.distance_cache.find_distance(
            self.controller.last_event.events[agent_id].metadata["agent"]["position"],
            target,
            retry_dist,
        )

    def agent_state(self, agent_id: int = 0) -> Dict:
        """Return agent position, rotation and horizon."""
        assert 0 <= agent_id < self.agent_count

        agent_meta = self.last_event.events[agent_id].metadata["agent"]
        return {
            **{k: float(v) for k, v in agent_meta["position"].items()},
            "rotation": {k: float(v) for k, v in agent_meta["rotation"].items()},
            "horizon": round(float(agent_meta["cameraHorizon"]), 1),
        }

    def teleport(
        self,
        pose: Dict[str, float],
        rotation: Dict[str, float],
        horizon: float = 0.0,
        agent_id: int = 0,
    ):
        assert 0 <= agent_id < self.agent_count
        e = self.controller.step(
            action="TeleportFull",
            x=pose["x"],
            y=pose["y"],
            z=pose["z"],
            rotation=rotation,
            horizon=horizon,
            standing=True,
            agentId=agent_id,
        )
        return e.metadata["lastActionSuccess"]

    def reset(
        self, scene_name: str = None, filtered_objects: Optional[List[str]] = None
    ) -> None:
        """Resets scene to a known initial state."""
        if scene_name is not None and scene_name != self.scene_name:
            self.controller.reset(scene_name)
            assert self.last_action_success, "Could not reset to new scene"

            if self.all_metadata_available and scene_name not in self.scene_to_reachable_positions:
                self.scene_to_reachable_positions[scene_name] = copy.deepcopy(
                    self.currently_reachable_points
                )
                assert len(self.scene_to_reachable_positions[scene_name]) > 10
        if filtered_objects:
            self.set_object_filter(filtered_objects)
        else:
            self.reset_object_filter()

    def random_reachable_state(
        self, seed: Optional[int] = None
    ) -> Dict[str, Union[Dict[str, float], float]]:
        """Returns a random reachable location in the scene."""
        assert self.all_metadata_available, (
            "`random_reachable_state` cannot be called when `self.all_metadata_available` is `False`."
        )

        if seed is not None:
            random.seed(seed)
        # xyz = random.choice(self.currently_reachable_points)
        assert len(self.scene_to_reachable_positions[self.scene_name]) > 10
        xyz = copy.deepcopy(random.choice(self.scene_to_reachable_positions[self.scene_name]))
        rotation = random.choice(
            np.arange(0.0, 360.0, self.config["rotateStepDegrees"])
        )
        horizon = 0.0  # random.choice([0.0, 30.0, 330.0])
        return {
            **{k: float(v) for k, v in xyz.items()},
            "rotation": {"x": 0.0, "y": float(rotation), "z": 0.0},
            "horizon": float(horizon),
        }

    def randomize_agent_location(
        self,
        seed: int = None,
        partial_position: Optional[Dict[str, float]] = None,
        agent_id: int = 0,
    ) -> Dict[str, Union[Dict[str, float], float]]:
        """Teleports the agent to a random reachable location in the scene."""
        assert 0 <= agent_id < self.agent_count

        if partial_position is None:
            partial_position = {}
        k = 0
        state: Optional[Dict] = None

        while k == 0 or (not self.last_action_success and k < 10):
            # self.reset()
            state = {**self.random_reachable_state(seed=seed), **partial_position}
            # get_logger().debug("picked target location {}".format(state))
            self.controller.step("TeleportFull", **state, agentId=agent_id)
            k += 1

        if not self.last_action_success:
            get_logger().warning(
                (
                    "Randomize agent location in scene {} and current random state {}"
                    " with seed {} and partial position {} failed in "
                    "10 attempts. Forcing the action."
                ).format(self.scene_name, state, seed, partial_position)
            )
            self.controller.step("TeleportFull", **state, force_action=True, agentId=agent_id)  # type: ignore
            assert self.last_action_success, "Force action failed with {}".format(state)

        # get_logger().debug("location after teleport full {}".format(self.agent_state()))
        # self.controller.step("TeleportFull", **self.agent_state())  # TODO only for debug
        # get_logger().debug("location after re-teleport full {}".format(self.agent_state()))

        return self.agent_state(agent_id=agent_id)

    def known_good_locations_list(self):
        assert self.all_metadata_available, (
            "`known_good_locations_list` cannot be called when `self.all_metadata_available` is `False`."
        )
        return self.scene_to_reachable_positions[self.scene_name]

    @property
    def currently_reachable_points(self) -> List[Dict[str, float]]:
        """List of {"x": x, "y": y, "z": z} locations in the scene that are
        currently reachable."""
        self.controller.step(action="GetReachablePositions")
        assert (
            self.last_action_success
        ), f"Could not get reachable positions for reason {self.last_event.metadata['errorMessage']}."
        return self.last_action_return

    @property
    def scene_name(self) -> str:
        """Current ai2thor scene."""
        return self.controller.last_event.metadata["sceneName"].replace("_physics", "")

    @property
    def current_frame(self) -> np.ndarray:
        """Returns rgb image corresponding to the agent's egocentric view."""
        return self.controller.last_event.frame

    @property
    def current_depth(self) -> np.ndarray:
        """Returns depth image corresponding to the agent's egocentric view."""
        return self.controller.last_event.depth_frame

    @property
    def current_frames(self) -> List[np.ndarray]:
        """Returns rgb images corresponding to the agents' egocentric views."""
        return [
            self.controller.last_event.events[agent_id].frame
            for agent_id in range(self.agent_count)
        ]

    @property
    def current_depths(self) -> List[np.ndarray]:
        """Returns depth images corresponding to the agents' egocentric
        views."""
        return [
            self.controller.last_event.events[agent_id].depth_frame
            for agent_id in range(self.agent_count)
        ]

    @property
    def last_event(self) -> ai2thor.server.Event:
        """Last event returned by the controller."""
        return self.controller.last_event

    @property
    def last_action(self) -> str:
        """Last action, as a string, taken by the agent."""
        return self.controller.last_event.metadata["lastAction"]

    @property
    def last_action_success(self) -> bool:
        """Was the last action taken by the agent a success?"""
        return self.controller.last_event.metadata["lastActionSuccess"]

    @property
    def last_action_return(self) -> Any:
        """Get the value returned by the last action (if applicable).

        For an example of an action that returns a value, see
        `"GetReachablePositions"`.
        """
        return self.controller.last_event.metadata["actionReturn"]

    def step(self, action_dict: Dict) -> ai2thor.server.Event:
        """Take a step in the ai2thor environment."""
        return self.controller.step(**action_dict)

    def stop(self):
        """Stops the ai2thor controller."""
        try:
            self.controller.stop()
        except Exception as e:
            get_logger().warning(str(e))

    def all_objects(self) -> List[Dict[str, Any]]:
        """Return all object metadata."""
        return self.controller.last_event.metadata["objects"]

    def all_objects_with_properties(
        self, properties: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Find all objects with the given properties."""
        objects = []
        for o in self.all_objects():
            satisfies_all = True
            for k, v in properties.items():
                if o[k] != v:
                    satisfies_all = False
                    break
            if satisfies_all:
                objects.append(o)
        return objects

    def visible_objects(self) -> List[Dict[str, Any]]:
        """Return all visible objects."""
        return self.all_objects_with_properties({"visible": True})


class RoboThorCachedEnvironment:
    """Wrapper for the robo2thor controller providing additional functionality
    and bookkeeping.

    See [here](https://ai2thor.allenai.org/robothor/documentation) for comprehensive
     documentation on RoboTHOR.

    # Attributes

    controller : The AI2THOR controller.
    config : The AI2THOR controller configuration
    """

    def __init__(self, **kwargs):
        self.config = dict(
            rotateStepDegrees=30.0,
            visibilityDistance=1.0,
            gridSize=0.25,
            # agentType="stochastic",
            continuousMode=True,
            snapToGrid=False,
            agentMode="bot",
            width=640,
            height=480,
        )
        self.env_root_dir = kwargs["env_root_dir"]
        random_scene = random.choice(list(glob.glob(self.env_root_dir + "/*.pkl")))
        handle = open(random_scene, "rb")
        self.view_cache = pickle.load(handle)
        handle.close()
        self.agent_position = list(self.view_cache.keys())[0]
        self.agent_rotation = list(self.view_cache[self.agent_position].keys())[0]
        self.known_good_locations: Dict[str, Any] = {
            self.scene_name: copy.deepcopy(self.currently_reachable_points)
        }
        self._last_action = "None"
        assert len(self.known_good_locations[self.scene_name]) > 10

    def agent_state(self) -> Dict[str, Union[Dict[str, float], float]]:
        """Return agent position, rotation and horizon."""
        return {
            **str_to_pos_for_cache(self.agent_position),
            "rotation": {"x": 0.0, "y": self.agent_rotation, "z": 0.0},
            "horizon": 1.0,
        }

    def teleport(
        self, pose: Dict[str, float], rotation: Dict[str, float], horizon: float = 0.0
    ):
        self.agent_position = pos_to_str_for_cache(pose)
        self.agent_rotation = (
            math.floor(rotation["y"] / 90.0) * 90
        )  # round to nearest 90 degree angle
        return True

    def reset(self, scene_name: str = None) -> None:
        """Resets scene to a known initial state."""
        try:
            handle = open(self.env_root_dir + "/" + scene_name + ".pkl", "rb")
            self.view_cache = pickle.load(handle)
            handle.close()
            self.agent_position = list(self.view_cache.keys())[0]
            self.agent_rotation = list(self.view_cache[self.agent_position].keys())[0]
            self.known_good_locations[self.scene_name] = copy.deepcopy(
                self.currently_reachable_points
            )
            self._last_action = "None"
            assert len(self.known_good_locations[self.scene_name]) > 10
        except Exception as _:
            raise RuntimeError("Could not load scene:", scene_name)

    def known_good_locations_list(self):
        return self.known_good_locations[self.scene_name]

    @property
    def currently_reachable_points(self) -> List[Dict[str, float]]:
        """List of {"x": x, "y": y, "z": z} locations in the scene that are
        currently reachable."""
        return [str_to_pos_for_cache(pos) for pos in self.view_cache]

    @property
    def scene_name(self) -> str:
        """Current ai2thor scene."""
        return self.view_cache[self.agent_position][self.agent_rotation].metadata[
            "sceneName"
        ]

    @property
    def current_frame(self) -> np.ndarray:
        """Returns rgb image corresponding to the agent's egocentric view."""
        return self.view_cache[self.agent_position][self.agent_rotation].frame

    @property
    def current_depth(self) -> np.ndarray:
        """Returns depth image corresponding to the agent's egocentric view."""
        return self.view_cache[self.agent_position][self.agent_rotation].depth_frame

    @property
    def last_event(self) -> ai2thor.server.Event:
        """Last event returned by the controller."""
        return self.view_cache[self.agent_position][self.agent_rotation]

    @property
    def last_action(self) -> str:
        """Last action, as a string, taken by the agent."""
        return self._last_action

    @property
    def last_action_success(self) -> bool:
        """In the cached environment, all actions succeed."""
        return True

    @property
    def last_action_return(self) -> Any:
        """Get the value returned by the last action (if applicable).

        For an example of an action that returns a value, see
        `"GetReachablePositions"`.
        """
        return self.view_cache[self.agent_position][self.agent_rotation].metadata[
            "actionReturn"
        ]

    def step(
        self, action_dict: Dict[str, Union[str, int, float]]
    ) -> ai2thor.server.Event:
        """Take a step in the ai2thor environment."""
        self._last_action = action_dict["action"]
        if action_dict["action"] == "RotateLeft":
            self.agent_rotation = (self.agent_rotation - 90.0) % 360.0
        elif action_dict["action"] == "RotateRight":
            self.agent_rotation = (self.agent_rotation + 90.0) % 360.0
        elif action_dict["action"] == "MoveAhead":
            pos = str_to_pos_for_cache(self.agent_position)
            if self.agent_rotation == 0.0:
                pos["x"] += 0.25
            elif self.agent_rotation == 90.0:
                pos["z"] += 0.25
            elif self.agent_rotation == 180.0:
                pos["x"] -= 0.25
            elif self.agent_rotation == 270.0:
                pos["z"] -= 0.25
            pos_string = pos_to_str_for_cache(pos)
            if pos_string in self.view_cache:
                self.agent_position = pos_to_str_for_cache(pos)
        return self.last_event

    # noinspection PyMethodMayBeStatic
    def stop(self):
        """Stops the ai2thor controller."""
        print("No need to stop cached environment")

    def all_objects(self) -> List[Dict[str, Any]]:
        """Return all object metadata."""
        return self.view_cache[self.agent_position][self.agent_rotation].metadata[
            "objects"
        ]

    def all_objects_with_properties(
        self, properties: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Find all objects with the given properties."""
        objects = []
        for o in self.all_objects():
            satisfies_all = True
            for k, v in properties.items():
                if o[k] != v:
                    satisfies_all = False
                    break
            if satisfies_all:
                objects.append(o)
        return objects

    def visible_objects(self) -> List[Dict[str, Any]]:
        """Return all visible objects."""
        return self.all_objects_with_properties({"visible": True})
