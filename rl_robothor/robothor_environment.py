import typing
from typing import Any, Optional, Dict, List, Union, Iterable, Tuple, Sized, Collection
import random
import copy
from collections import OrderedDict
import math
import pickle
import glob

import ai2thor
from ai2thor.controller import Controller
from ai2thor.util import metrics
import numpy as np

from utils.experiment_utils import recursive_update
from utils.cache_utils import _str_to_pos, _pos_to_str
from utils.system import LOGGER


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
        self.known_good_locations: Dict[str, Any] = {self.scene_name: copy.deepcopy(self.currently_reachable_points)}
        assert len(self.known_good_locations[self.scene_name]) > 50

        # onames = [o['objectId'] for o in self.last_event.metadata['objects']]
        # removed = []
        # for oname in onames:
        #     if 'Painting' in oname:
        #         self.controller.step("RemoveFromScene", objectId=oname)
        #         removed.append(oname)
        # LOGGER.info("Removed {} Paintings from {}".format(len(removed), self.scene_name))

        # LOGGER.warning("init to scene {} in pos {}".format(self.scene_name, self.agent_state()))
        # npoints = len(self.currently_reachable_points)
        # assert npoints > 100, "only {} reachable points after init".format(npoints)
        self.grids: Dict[str, Tuple[Dict[str, np.array], int, int, int, int]] = {}
        self.initialize_grid()

    def initialize_grid_dimensions(
        self, reachable_points: Collection[Dict[str, float]]
    ) -> Tuple[int, int, int, int]:
        """Computes bounding box for reachable points quantized with the current gridSize."""
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

    def access_grid(self, target: str) -> float:
        """Returns the geodesic distance from the quantized location of the agent in the current scene's grid to the
         target object of given type."""
        if target not in self.grids[self.scene_name][0]:
            xmin, xmax, zmin, zmax = self.grids[self.scene_name][1:5]
            nx = xmax - xmin + 1
            nz = zmax - zmin + 1
            self.grids[self.scene_name][0][target] = -2 * np.ones(
                (nx, nz), dtype=np.float64
            )

        p = self.quantized_agent_state()

        if self.grids[self.scene_name][0][target][p[0], p[1]] < -1.5:
            corners = self.path_corners(target)
            dist = self.path_corners_to_dist(corners)
            if dist == float("inf"):
                dist = -1.0  # -1.0 for unreachable
            self.grids[self.scene_name][0][target][p[0], p[1]] = dist
            return dist

        return self.grids[self.scene_name][0][target][p[0], p[1]]

    def initialize_grid(self) -> None:
        """Initializes grid for current scene if not already initialized."""
        if self.scene_name in self.grids:
            return

        self.grids[self.scene_name] = ({},) + self.initialize_grid_dimensions(self.known_good_locations[self.scene_name])  # type: ignore

    def object_reachable(self, object_type: str) -> bool:
        """Determines whether a path can be computed from the discretized current agent location to the target object
         of given type."""
        return (
            self.access_grid(object_type) > -0.5
        )  # -1.0 for unreachable, 0.0 for end point

    def point_reachable(self, xyz: Dict[str, float]) -> bool:
        """Determines whether a path can be computed from the current agent location to the target point."""
        return (
            self.dist_to_point(xyz) > -0.5
        )  # -1.0 for unreachable, 0.0 for end point

    def path_corners(self, target: Union[str, Dict[str, float]]) -> Collection[Dict[str, float]]:
        """Returns an array with a sequence of xyz dictionaries objects representing the corners of the shortest path
         to the object of given type or end point location."""
        pose = self.agent_state()
        position = {k: float(pose[k]) for k in ["x", "y", "z"]}
        # LOGGER.debug("initial pos in path corners {} target {}".format(pose, target))
        try:
            if isinstance(target, str):
                path = metrics.get_shortest_path_to_object_type(
                    self.controller,
                    target,
                    position,
                    {**pose["rotation"]} if "rotation" in pose else None,
                )
            else:
                path = metrics.get_shortest_path_to_point(self.controller, position, target)
        except ValueError:
            LOGGER.debug("No path to object {} from {} in {}".format(target, position, self.scene_name))
            path = []
        finally:
            if isinstance(target, str):
                self.controller.step("TeleportFull", **pose)
                # pass
            new_pose = self.agent_state()
            try:
                assert abs(new_pose['x'] - pose['x']) < 1e-5, "wrong x"
                assert abs(new_pose['y'] - pose['y']) < 1e-5, "wrong y"
                assert abs(new_pose['z'] - pose['z']) < 1e-5, "wrong z"
                assert abs(new_pose['rotation']['x'] - pose['rotation']['x']) < 1e-5, "wrong rotation x"
                assert abs(new_pose['rotation']['y'] - pose['rotation']['y']) < 1e-5, "wrong rotation y"
                assert abs(new_pose['rotation']['z'] - pose['rotation']['z']) < 1e-5, "wrong rotation z"
                assert abs((new_pose['horizon'] % 360) - (pose['horizon'] % 360)) < 1e-5, "wrong horizon {} vs {}".format((new_pose['horizon'] % 360), (pose['horizon'] % 360))
            except Exception:
                LOGGER.error("new_pose {} old_pose {} in {}".format(new_pose, pose, self.scene_name))
            # if abs((new_pose['horizon'] % 360) - (pose['horizon'] % 360)) > 1e-5:
            #     LOGGER.debug("wrong horizon {} vs {} after path to object {} from {} in {}".format((new_pose['horizon'] % 360), (pose['horizon'] % 360), target, position, self.scene_name))
            # else:
            #     LOGGER.debug("correct horizon {} vs {} after path to object {} from {} in {}".format((new_pose['horizon'] % 360), (pose['horizon'] % 360), target, position, self.scene_name))
            # assert abs((new_pose['horizon'] % 360) - (pose['horizon'] % 360)) < 1e-5, "wrong horizon {} vs {}".format((new_pose['horizon'] % 360), (pose['horizon'] % 360))

            # # TODO: the agent will continue with a random horizon from here on
            # target_horizon = (pose['horizon'] % 360) - (360 if (pose['horizon'] % 360) >= 180 else 0)
            # new_pose = self.agent_state()['horizon']
            # update_horizon = (new_pose % 360) - (360 if (new_pose % 360) >= 180 else 0)
            # cond = abs(target_horizon - update_horizon) > 1e-5
            # nmovements = 0
            # while cond:
            #     cond = abs(target_horizon - update_horizon) > 1e-5 and target_horizon > update_horizon
            #     while cond:
            #         self.controller.step("LookDown")
            #         old = update_horizon
            #         new_pose = self.agent_state()['horizon']
            #         update_horizon = (new_pose % 360) - (360 if (new_pose % 360) >= 180 else 0)
            #         LOGGER.debug("LookDown horizon {} -> {} ({})".format(old, update_horizon, target_horizon))
            #         nmovements += 1
            #         cond = abs(target_horizon - update_horizon) > 1e-5 and target_horizon > update_horizon
            #
            #     cond = abs(target_horizon - update_horizon) > 1e-5 and target_horizon < update_horizon
            #     while cond:
            #         self.controller.step("LookUp")
            #         old = update_horizon
            #         new_pose = self.agent_state()['horizon']
            #         update_horizon = (new_pose % 360) - (360 if (new_pose % 360) >= 180 else 0)
            #         LOGGER.debug("LookUp horizon {} -> {} ({})".format(old, update_horizon, target_horizon))
            #         nmovements += 1
            #         cond = abs(target_horizon - update_horizon) > 1e-5 and target_horizon < update_horizon
            #
            #     cond = abs(target_horizon - update_horizon) > 1e-5
            # LOGGER.debug("nmovements {}".format(nmovements))
            # new_pose = self.agent_state()
            # assert abs((new_pose['horizon'] % 360) - (pose['horizon'] % 360)) < 1e-5, "wrong horizon {} vs {}".format((new_pose['horizon'] % 360), (pose['horizon'] % 360))

            # try:
            #     assert abs((new_pose['horizon'] % 360) - (pose['horizon'] % 360)) < 1e-5, "wrong horizon {} vs {}".format((new_pose['horizon'] % 360), (pose['horizon'] % 360))
            # except Exception:
            #     LOGGER.error("wrong horizon {} vs {}".format((new_pose['horizon'] % 360), (pose['horizon'] % 360)))
            #     self.controller.step("TeleportFull", **pose)
            #     assert abs(
            #         (new_pose['horizon'] % 360) - (pose['horizon'] % 360)) < 1e-5, "wrong horizon {} vs {} after teleport full".format(
            #         (new_pose['horizon'] % 360), (pose['horizon'] % 360))
        #     # LOGGER.debug("initial pos in path corners {} current pos {} path {}".format(pose, self.agent_state(), path))
        return path

    def path_corners_to_dist(self, corners: Collection[Dict[str, float]]) -> float:
        """Computes the distance covered by the given path described by its corners."""

        if len(corners) == 0:
            return float("inf")

        sum = 0
        for it in range(1, len(corners)):
            sum += math.sqrt(
                (corners[it]["x"] - corners[it - 1]["x"]) ** 2
                + (corners[it]["z"] - corners[it - 1]["z"]) ** 2
            )
        return sum

    def quantized_agent_state(
        self, xz_subsampling: int = 1, rot_subsampling: int = 1
    ) -> Tuple[int, int, int]:
        """Quantizes agent location (x, z) to a (subsampled) position in a fixed size grid derived from the initial set
         of reachable points; and rotation (around y axis) as a (subsampled) discretized angle given the current
          `rotateStepDegrees`."""
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

    def dist_to_object(self, object_type: str) -> float:
        """Minimal geodesic distance to object of given type from agent's current location. It might return -1.0 for
         unreachable targets."""
        return self.access_grid(object_type)

    def dist_to_point(self, xyz: Dict[str, float]) -> float:
        """Minimal geodesic distance to end point from agent's current location. It might return -1.0 for
         unreachable targets."""
        corners = self.path_corners(xyz)
        dist = self.path_corners_to_dist(corners)
        if dist == float("inf"):
            dist = -1.0  # -1.0 for unreachable
        return dist

    def agent_state(self) -> Dict[str, Union[Dict[str, float], float]]:
        """Return agent position, rotation and horizon."""
        agent_meta = self.last_event.metadata["agent"]
        return {
            **{k: float(v) for k, v in agent_meta["position"].items()},
            "rotation": {k: float(v) for k, v in agent_meta["rotation"].items()},
            "horizon": round(float(agent_meta["cameraHorizon"]), 1),
        }

    def teleport(self, pose: Dict[str,float], rotation: Dict[str,float], horizon: float=0.0):
        e = self.controller.step("TeleportFull",
                                 x=pose["x"],
                                 y=pose["y"],
                                 z=pose["z"],
                                 rotation=rotation,
                                 horizon=horizon)
        return e.metadata["lastActionSuccess"]

    def reset(self, scene_name: str = None) -> None:
        """Resets scene to a known initial state."""
        if scene_name is not None and scene_name != self.scene_name:
            self.controller.reset(scene_name)
            assert self.last_action_success, "Could not reset to new scene"
            if scene_name not in self.known_good_locations:
                self.known_good_locations[scene_name] = copy.deepcopy(self.currently_reachable_points)
                assert len(self.known_good_locations[scene_name]) > 50

            # onames = [o['objectId'] for o in self.last_event.metadata['objects']]
            # removed = []
            # for oname in onames:
            #     if 'Painting' in oname:
            #         self.controller.step("RemoveFromScene", objectId=oname)
            #         removed.append(oname)
            # LOGGER.info("Removed {} Paintings from {}".format(len(removed), scene_name))

        # else:
            # assert (
            #     self.scene_name in self.known_good_locations
            # ), "Resetting scene without known good location"
            # LOGGER.warning("Resetting {} to {}".format(self.scene_name, self.known_good_locations[self.scene_name]))
            # self.controller.step("TeleportFull", **self.known_good_locations[self.scene_name])
            # assert self.last_action_success, "Could not reset to known good location"

        # npoints = len(self.currently_reachable_points)
        # assert npoints > 100, "only {} reachable points after reset".format(npoints)

        self.initialize_grid()

    def randomize_agent_location(
        self, seed: int = None, partial_position: Optional[Dict[str, float]] = None
    ) -> Dict[str, Union[Dict[str, float], float]]:
        """Teleports the agent to a random reachable location in the scene."""
        if partial_position is None:
            partial_position = {}
        k = 0
        state: Optional[Dict] = None

        while k == 0 or (not self.last_action_success and k < 10):
            # self.reset()
            state = {**self.random_reachable_state(seed=seed), **partial_position}
            # LOGGER.debug("picked target location {}".format(state))
            self.controller.step("TeleportFull", **state)
            k += 1

        if not self.last_action_success:
            LOGGER.warning(
                (
                    "Randomize agent location in scene {} and current random state {}"
                    " with seed {} and partial position {} failed in "
                    "10 attempts. Forcing the action."
                ).format(self.scene_name, state, seed, partial_position)
            )
            self.controller.step("TeleportFull", **state, force_action=True)  # type: ignore
            assert self.last_action_success, "Force action failed with {}".format(state)

        # LOGGER.debug("location after teleport full {}".format(self.agent_state()))
        # self.controller.step("TeleportFull", **self.agent_state())  # TODO only for debug
        # LOGGER.debug("location after re-teleport full {}".format(self.agent_state()))

        return self.agent_state()

    def random_reachable_state(
        self, seed: Optional[int] = None
    ) -> Dict[str, Union[Dict[str, float], float]]:
        """Returns a random reachable location in the scene."""
        if seed is not None:
            random.seed(seed)
        # xyz = random.choice(self.currently_reachable_points)
        assert len(self.known_good_locations[self.scene_name]) > 100
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

    def step(
        self, action_dict: Dict[str, Union[str, int, float]]
    ) -> ai2thor.server.Event:
        """Take a step in the ai2thor environment."""
        return self.controller.step(**action_dict)

    def stop(self):
        """Stops the ai2thor controller."""
        try:
            self.controller.stop()
        except Exception as e:
            LOGGER.warning(str(e))

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


# def get_shortest_path_to_point(
#         controller,
#         initial_position,
#         target_position
# ):
#     """
#     Computes the shortest path to an end point from an initial position using a controller
#     :param controller: agent controller
#     :param initial_position: dict(x=float, y=float, z=float) with the desired initial position
#     :param target_position: dict(x=float, y=float, z=float) with the desired target position
#     """
#     args = dict(
#         action='GetShortestPathToPoint',
#         position=initial_position,
#         x=target_position['x'],
#         y=target_position['y'],
#         z=target_position['z']
#     )
#     event = controller.step(args)
#     if event.metadata['lastActionSuccess']:
#         return event.metadata['actionReturn']['corners']
#     else:
#         raise ValueError(
#             "Unable to find shortest path for target point '{}'".format(
#                 target_position
#             )
#         )

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
        handle= open(random_scene, 'rb')
        self.view_cache = pickle.load(handle)
        handle.close()
        self.agent_position = list(self.view_cache.keys())[0]
        self.agent_rotation = list(self.view_cache[self.agent_position].keys())[0]
        self.known_good_locations: Dict[str, Any] = {self.scene_name: copy.deepcopy(self.currently_reachable_points)}
        self._last_action = "None"
        assert len(self.known_good_locations[self.scene_name]) > 100

    def agent_state(self) -> Dict[str, Union[Dict[str, float], float]]:
        """Return agent position, rotation and horizon."""
        return {
            **_str_to_pos(self.agent_position),
            "rotation": self.agent_rotation,
            "horizon": 1.0,
        }

    def teleport(self, pose: Dict[str,float], rotation: Dict[str,float], horizon: float=0.0):
        self.agent_position = _pos_to_str(pose)
        self.agent_rotation = math.floor(rotation / 90.0) * 90 # round to nearest 90 degree angle
        return True

    def reset(self, scene_name: str = None) -> None:
        """Resets scene to a known initial state."""
        try:
            handle = open(self.env_root_dir + "/" + scene_name + ".pkl", 'rb')
            self.view_cache = pickle.load(handle)
            handle.close()
            self.agent_position = list(self.view_cache.keys())[0]
            self.agent_rotation = list(self.view_cache[self.agent_position].keys())[0]
            self.known_good_locations: Dict[str, Any] = {
                self.scene_name: copy.deepcopy(self.currently_reachable_points)
            }
            self._last_action = "None"
            assert len(self.known_good_locations[self.scene_name]) > 100
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
        return self.view_cache[self.agent_position][self.agent_rotation].metadata["sceneName"]

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
        """In the cached environment, all actions succeed"""
        return True

    @property
    def last_action_return(self) -> Any:
        """Get the value returned by the last action (if applicable).

        For an example of an action that returns a value, see
        `"GetReachablePositions"`.
        """
        return self.view_cache[self.agent_position][self.agent_rotation].metadata["actionReturn"]

    def step(
        self, action_dict: Dict[str, Union[str, int, float]]
    ) -> ai2thor.server.Event:
        """Take a step in the ai2thor environment."""
        self._last_action = action_dict['action']
        if action_dict['action'] == 'RotateLeft':
            self.agent_rotation = (self.agent_rotation - 90.0) % 360.0
        elif action_dict['action'] == "RotateRight":
            self.agent_rotation = (self.agent_rotation + 90.0) % 360.0
        elif action_dict['action'] == "MoveAhead":
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
        return self.view_cache[self.agent_position][self.agent_rotation].metadata["objects"]

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
