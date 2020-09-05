import copy
import glob
import math
import pickle
import random
from typing import Any, Optional, Dict, List, Union, Tuple, Collection, Sequence

import ai2thor.server
import numpy as np
from ai2thor.controller import Controller
from ai2thor.util import metrics

from utils.cache_utils import _str_to_pos, _pos_to_str
from utils.experiment_utils import recursive_update
from utils.system import get_logger


class RoboThorEnvironment:
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
            agentType="stochastic",
            continuousMode=True,
            snapToGrid=False,
            agentMode="bot",
            width=640,
            height=480,
        )
        recursive_update(self.config, {**kwargs, "agentMode": "bot"})
        self.controller = Controller(**self.config)
        self.known_good_locations: Dict[str, Any] = {
            self.scene_name: copy.deepcopy(self.currently_reachable_points)
        }
        assert len(self.known_good_locations[self.scene_name]) > 10

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

    def object_reachable(self, object_type: str) -> bool:
        """Determines whether a path can be computed from the discretized
        current agent location to the target object of given type."""
        return (
            self.access_grid(object_type) > -0.5
        )  # -1.0 for unreachable, 0.0 for end point

    def point_reachable(self, xyz: Dict[str, float]) -> bool:
        """Determines whether a path can be computed from the current agent
        location to the target point."""
        return self.dist_to_point(xyz) > -0.5  # -1.0 for unreachable, 0.0 for end point

    def quantized_agent_state(
        self, xz_subsampling: int = 1, rot_subsampling: int = 1
    ) -> Tuple[int, int, int]:
        """Quantizes agent location (x, z) to a (subsampled) position in a
        fixed size grid derived from the initial set of reachable points; and
        rotation (around y axis) as a (subsampled) discretized angle given the
        current `rotateStepDegrees`."""
        pose = self.agent_state()
        p = {k: float(pose[k]) for k in ["x", "y", "z"]}

        xmin, xmax, zmin, zmax = self.grids[self.scene_name][1:5]
        x = int(np.clip(round(p["x"] / self.config["gridSize"]), xmin, xmax))
        z = int(np.clip(round(p["z"] / self.config["gridSize"]), zmin, zmax))

        rs = self.config["rotateStepDegrees"] * rot_subsampling
        shifted = pose["rotation"]["y"] + rs / 2
        normalized = shifted % 360.0
        r = int(round(normalized / rs))

        return (x - xmin) // xz_subsampling, (z - zmin) // xz_subsampling, r

    def path_to_object_type(self, object_type: str) -> List[Dict[str, float]]:
        try:
            return metrics.get_shortest_path_to_object_type(
                self.controller,
                object_type,
                self.controller.last_event.metadata["agent"]["position"]
            )
        except:
            return None

    def distance_to_object_type(self, object_type: str) -> float:
        """Minimal geodesic distance to object of given type from agent's
        current location.

        It might return -1.0 for unreachable targets.
        """
        path = self.path_to_object_type(object_type)
        if path:
            return metrics.path_distance(path)
        return -1.0

    def dist_to_point(self, xyz: Dict[str, float]) -> float:
        """Minimal geodesic distance to end point from agent's current
        location.

        It might return -1.0 for unreachable targets.
        """
        # TODO: MAke sure this method gets implemented in thor
        raise NotImplementedError

    def agent_state(self) -> Dict:
        """Return agent position, rotation and horizon."""
        agent_meta = self.last_event.metadata["agent"]
        return {
            **{k: float(v) for k, v in agent_meta["position"].items()},
            "rotation": {k: float(v) for k, v in agent_meta["rotation"].items()},
            "horizon": round(float(agent_meta["cameraHorizon"]), 1),
        }

    def teleport(
        self, pose: Dict[str, float], rotation: Dict[str, float], horizon: float = 0.0
    ):
        e = self.controller.step(
            "TeleportFull",
            x=pose["x"],
            y=pose["y"],
            z=pose["z"],
            rotation=rotation,
            horizon=horizon,
        )
        return e.metadata["lastActionSuccess"]

    def reset(self, scene_name: str = None) -> None:
        """Resets scene to a known initial state."""
        if scene_name is not None and scene_name != self.scene_name:
            self.controller.reset(scene_name)
            assert self.last_action_success, "Could not reset to new scene"
            if scene_name not in self.known_good_locations:
                self.known_good_locations[scene_name] = copy.deepcopy(
                    self.currently_reachable_points
                )
                assert len(self.known_good_locations[scene_name]) > 10


    def random_reachable_state(
        self, seed: Optional[int] = None
    ) -> Dict[str, Union[Dict[str, float], float]]:
        """Returns a random reachable location in the scene."""
        if seed is not None:
            random.seed(seed)
        # xyz = random.choice(self.currently_reachable_points)
        assert len(self.known_good_locations[self.scene_name]) > 10
        xyz = copy.deepcopy(random.choice(self.known_good_locations[self.scene_name]))
        rotation = random.choice(
            np.arange(0.0, 360.0, self.config["rotateStepDegrees"])
        )
        horizon = 0.0  # random.choice([0.0, 30.0, 330.0])
        return {
            **{k: float(v) for k, v in xyz.items()},
            "rotation": {"x": 0.0, "y": float(rotation), "z": 0.0},
            "horizon": float(horizon),
        }

    def known_good_locations_list(self):
        return self.known_good_locations[self.scene_name]

    @property
    def currently_reachable_points(self) -> List[Dict[str, float]]:
        """List of {"x": x, "y": y, "z": z} locations in the scene that are
        currently reachable."""
        self.controller.step(action="GetReachablePositions")
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
            **_str_to_pos(self.agent_position),
            "rotation": {"x": 0.0, "y": self.agent_rotation, "z": 0.0},
            "horizon": 1.0,
        }

    def teleport(
        self, pose: Dict[str, float], rotation: Dict[str, float], horizon: float = 0.0
    ):
        self.agent_position = _pos_to_str(pose)
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
        except:
            print("Could not load scene:", scene_name)

    def known_good_locations_list(self):
        return self.known_good_locations[self.scene_name]

    @property
    def currently_reachable_points(self) -> List[Dict[str, float]]:
        """List of {"x": x, "y": y, "z": z} locations in the scene that are
        currently reachable."""
        return [_str_to_pos(pos) for pos in self.view_cache]

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
            pos = _str_to_pos(self.agent_position)
            if self.agent_rotation == 0.0:
                pos["x"] += 0.25
            elif self.agent_rotation == 90.0:
                pos["z"] += 0.25
            elif self.agent_rotation == 180.0:
                pos["x"] -= 0.25
            elif self.agent_rotation == 270.0:
                pos["z"] -= 0.25
            pos_string = _pos_to_str(pos)
            if pos_string in self.view_cache:
                self.agent_position = _pos_to_str(pos)
        return True

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
