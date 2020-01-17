"""A wrapper for engaging with the THOR environment."""

import copy
import json
import math
import os
import random
import sys
import warnings
from typing import Tuple, Dict, List, Set, Union, Any, Optional, Mapping, NamedTuple

import ai2thor.server
import numpy as np
from ai2thor.controller import Controller

from extensions.ai2thor.constants import VISIBILITY_DISTANCE, FOV


class AI2ThorEnvironment(object):
    def __init__(
        self,
        docker_enabled: bool = False,
        x_display: str = None,
        local_thor_build: str = None,
        time_scale: float = 1.0,
        visibility_distance: float = VISIBILITY_DISTANCE,
        fov: float = FOV,
        player_screen_width: int = 300,
        player_screen_height: int = 300,
        quality: str = "Very Low",
        restrict_to_initially_reachable_points: bool = False,
        make_agents_visible: bool = True,
        object_open_speed: float = 1.0,
        always_return_visible_range: bool = False,
        simplify_physics: bool = False,
    ) -> None:
        self.controller = Controller()
        self.controller.local_executable_path = local_thor_build
        self.controller.docker_enabled = docker_enabled
        self.x_display = x_display
        self._initially_reachable_points: Optional[List[Dict]] = None
        self._initially_reachable_points_set: Optional[Set[Dict]] = None
        self._started = False
        self._offset: Optional[Tuple[int, int]] = None
        self.move_mag: Optional[float] = None
        self.grid_size: Optional[float] = None
        self.time_scale = time_scale
        self.visibility_distance = visibility_distance
        self.fov = fov
        self.restrict_to_initially_reachable_points = (
            restrict_to_initially_reachable_points
        )
        self.make_agents_visible = make_agents_visible
        self.object_open_speed = object_open_speed
        self.always_return_visible_range = always_return_visible_range
        self.simplify_physics = simplify_physics

        self._start_player_screen_width = player_screen_width
        self._start_player_screen_height = player_screen_height
        self._quality = quality

    @property
    def scene_name(self) -> str:
        return self.controller.last_event.metadata["sceneName"]

    @property
    def current_frame(self) -> np.ndarray:
        return self.controller.last_event.frame

    @property
    def offset(self) -> Tuple[int, int]:
        assert self._offset is not None
        return self._offset

    @property
    def last_event(self) -> ai2thor.server.Event:
        return self.controller.last_event

    @property
    def started(self) -> bool:
        return self._started

    @property
    def last_action(self):
        return self.controller.last_event.metadata["lastAction"]

    @last_action.setter
    def last_action(self, value: str):
        self.controller.last_event.metadata["lastAction"] = value

    @property
    def last_action_success(self):
        return self.controller.last_event.metadata["lastActionSuccess"]

    @last_action_success.setter
    def last_action_success(self, value: bool):
        self.controller.last_event.metadata["lastActionSuccess"] = value

    @property
    def last_action_return(self):
        return self.controller.last_event.metadata["actionReturn"]

    @last_action_return.setter
    def last_action_return(self, value: Any):
        self.controller.last_event.metadata["actionReturn"] = value

    def start(
        self,
        scene_name: Optional[str],
        move_mag: float = 0.25,
        offset: Tuple[int, int] = (0, 0),
        **kwargs,
    ) -> None:
        if (
            self._start_player_screen_width < 300
            or self._start_player_screen_height < 300
        ):
            self.controller.start(
                x_display=self.x_display,
                player_screen_width=300,
                player_screen_height=300,
            )
            self.controller.step(
                {
                    "action": "ChangeResolution",
                    "x": self._start_player_screen_width,
                    "y": self._start_player_screen_height,
                }
            )
        else:
            self.controller.start(
                x_display=self.x_display,
                player_screen_width=self._start_player_screen_width,
                player_screen_height=self._start_player_screen_height,
            )

        self.controller.step({"action": "ChangeQuality", "quality": self._quality})
        if not self.controller.last_event.metadata["lastActionSuccess"]:
            raise Exception("Failed to change quality to: {}.".format(self._quality))

        self._started = True
        self.reset(scene_name=scene_name, move_mag=move_mag, offset=offset, **kwargs)

    def stop(self) -> None:
        try:
            self.controller.stop()
        except Exception as e:
            warnings.warn(str(e))
        finally:
            self._started = False

    def reset(
        self,
        scene_name: Optional[str],
        move_mag: float = 0.25,
        offset: Tuple[int, int] = (0, 0),
        **kwargs,
    ):
        if offset != (0, 0):
            raise NotImplementedError("Offset must be (0,0) for now.")

        self._offset = offset
        self.move_mag = move_mag
        x_off = offset[0]
        z_off = offset[1]
        self.grid_size = self.move_mag * math.gcd(math.gcd(100, x_off), z_off) / 100.0

        if scene_name is None:
            scene_name = self.controller.last_event.metadata["sceneName"]
        self.controller.reset(scene_name)

        tmp_stderr = sys.stderr
        sys.stderr = open(
            os.devnull, "w"
        )  # TODO: HACKILY BLOCKING sequenceId print errors
        self.controller.step(
            {
                "action": "Initialize",
                "gridSize": self.grid_size,
                "visibilityDistance": self.visibility_distance,
                "fov": self.fov,
                "timeScale": self.time_scale,
                "makeAgentsVisible": self.make_agents_visible,
                "alwaysReturnVisibleRange": self.always_return_visible_range,
                **kwargs,
            }
        )
        sys.stderr.close()
        sys.stderr = tmp_stderr

        if self.object_open_speed != 1.0:
            self.controller.step(
                {"action": "ChangeOpenSpeed", "x": self.object_open_speed}
            )

        self._initially_reachable_points = None
        self._initially_reachable_points_set = None
        self.controller.step({"action": "GetReachablePositions"})
        if not self.controller.last_event.metadata["lastActionSuccess"]:
            warnings.warn(
                "Error when getting reachable points: {}".format(
                    self.controller.last_event.metadata["errorMessage"]
                )
            )
        self._initially_reachable_points = self.last_action_return

    def teleport_agent_to(
        self,
        x: float,
        y: float,
        z: float,
        rotation: float,
        horizon: float,
        standing: Optional[bool] = None,
        force_action: bool = False,
        only_initially_reachable: Optional[bool] = None,
        verbose=True,
        ignore_y_diffs=False,
    ) -> None:
        if standing is None:
            standing = self.last_event.metadata["isStanding"]
        original_location = self.get_agent_location()
        target = {"x": x, "y": y, "z": z}
        if only_initially_reachable is None:
            only_initially_reachable = self.restrict_to_initially_reachable_points
        if only_initially_reachable:
            reachable_points = self.initially_reachable_points
            reachable = False
            for p in reachable_points:
                if self.position_dist(target, p, ignore_y=ignore_y_diffs) < 0.01:
                    reachable = True
                    break
            if not reachable:
                self.last_action = "TeleportFull"
                self.last_event.metadata[
                    "errorMessage"
                ] = "Target position was not initially reachable."
                self.last_action_success = False
                return
        self.controller.step(
            dict(
                action="TeleportFull",
                x=x,
                y=y,
                z=z,
                rotation={"x": 0.0, "y": rotation, "z": 0.0},
                horizon=horizon,
                standing=standing,
                forceAction=force_action,
            )
        )
        if not self.last_action_success:
            agent_location = self.get_agent_location()
            rot_diff = (
                agent_location["rotation"] - original_location["rotation"]
            ) % 360
            new_old_dist = self.position_dist(
                original_location, agent_location, ignore_y=ignore_y_diffs
            )
            if (
                self.position_dist(
                    original_location, agent_location, ignore_y=ignore_y_diffs
                )
                > 1e-2
                or min(rot_diff, 360 - rot_diff) > 1
            ):
                print(x)
                warnings.warn(
                    "Teleportation FAILED but agent still moved (position_dist {}, rot diff {})"
                    " (\nprevious location\n{}\ncurrent_location\n{}\n)".format(
                        new_old_dist, rot_diff, original_location, agent_location
                    )
                )
            return

        if force_action:
            assert self.last_action_success
            return

        agent_location = self.get_agent_location()
        rot_diff = (agent_location["rotation"] - rotation) % 360
        if (
            self.position_dist(agent_location, target, ignore_y=ignore_y_diffs) > 1e-2
            or min(rot_diff, 360 - rot_diff) > 1
        ):
            if only_initially_reachable:
                self._snap_agent_to_initially_reachable(verbose=False)
            if verbose:
                warnings.warn(
                    "Teleportation did not place agent"
                    " precisely where desired in scene {}"
                    " (\ndesired\n{}\nactual\n{}\n)"
                    " perhaps due to grid snapping."
                    " Action is considered failed but agent may have moved.".format(
                        self.scene_name,
                        {
                            "x": x,
                            "y": y,
                            "z": z,
                            "rotation": rotation,
                            "standing": standing,
                            "horizon": horizon,
                        },
                        agent_location,
                    )
                )
            self.last_action_success = False
        return

    def random_reachable_state(self, seed: int = None) -> Dict:
        if seed is not None:
            random.seed(seed)
        xyz = random.choice(self.currently_reachable_points)
        rotation = random.choice([0, 90, 180, 270])
        horizon = random.choice([0, 30, 60, 330])
        state = copy.copy(xyz)
        state["rotation"] = rotation
        state["horizon"] = horizon
        return state

    def randomize_agent_location(
        self, seed: int = None, partial_position: Optional[Dict[str, float]] = None
    ) -> Dict:
        if partial_position is None:
            partial_position = {}
        k = 0
        state: Optional[Dict] = None

        while k == 0 or (not self.last_action_success and k < 10):
            state = self.random_reachable_state(seed=seed)
            self.teleport_agent_to(**{**state, **partial_position})
            k += 1

        if not self.last_action_success:
            warnings.warn(
                (
                    "Randomize agent location in scene {}"
                    " with seed {} and partial position {} failed in "
                    "10 attempts. Forcing the action."
                ).format(self.scene_name, seed, partial_position)
            )
            self.teleport_agent_to(**{**state, **partial_position}, force_action=True)
            assert self.last_action_success

        assert state is not None
        return state

    def object_pixels_in_frame(
        self, object_id: str, hide_all: bool = True, hide_transparent: bool = False
    ) -> np.ndarray:
        # Emphasizing an object turns it magenta and hides all other objects
        # from view, we can find where the hand object is on the screen by
        # emphasizing it and then scanning across the image for the magenta pixels.
        if hide_all:
            self.step({"action": "EmphasizeObject", "objectId": object_id})
        else:
            self.step({"action": "MaskObject", "objectId": object_id})
            if hide_transparent:
                self.step({"action": "HideTranslucentObjects"})
        filter = np.array([[[255, 0, 255]]])
        object_pixels = 1 * np.all(self.current_frame == filter, axis=2)
        if hide_all:
            self.step({"action": "UnemphasizeAll"})
        else:
            self.step({"action": "UnmaskObject", "objectId": object_id})
            if hide_transparent:
                self.step({"action": "UnhideAllObjects"})
        return object_pixels

    def object_pixels_on_grid(
        self,
        object_id: str,
        grid_shape: Tuple[int, int],
        hide_all: bool = True,
        hide_transparent: bool = False,
    ) -> np.ndarray:
        def partition(n, num_parts):
            m = n // num_parts
            parts = [m] * num_parts
            num_extra = n % num_parts
            for i in range(num_extra):
                parts[i] += 1
            return parts

        object_pixels = self.object_pixels_in_frame(
            object_id=object_id, hide_all=hide_all, hide_transparent=hide_transparent
        )

        # Divide the current frame into a grid and count the number
        # of hand object pixels in each of the grid squares
        sums_in_blocks: List[List] = []
        frame_shape = self.current_frame.shape[:2]
        row_inds = np.cumsum([0] + partition(frame_shape[0], grid_shape[0]))
        col_inds = np.cumsum([0] + partition(frame_shape[1], grid_shape[1]))
        for i in range(len(row_inds) - 1):
            sums_in_blocks.append([])
            for j in range(len(col_inds) - 1):
                sums_in_blocks[i].append(
                    np.sum(
                        object_pixels[
                            row_inds[i] : row_inds[i + 1], col_inds[j] : col_inds[j + 1]
                        ]
                    )
                )
        return np.array(sums_in_blocks, dtype=np.float32)

    def object_in_hand(self):
        inv_objs = self.last_event.metadata["inventoryObjects"]
        if len(inv_objs) == 0:
            return None
        elif len(inv_objs) == 1:
            return self.get_object_by_id(
                self.last_event.metadata["inventoryObjects"][0]["objectId"]
            )
        else:
            raise AttributeError("Must be <= 1 inventory objects.")

    @property
    def initially_reachable_points(self) -> List[Dict[str, float]]:
        assert self._initially_reachable_points is not None
        return copy.deepcopy(self._initially_reachable_points)  # type:ignore

    @property
    def initially_reachable_points_set(self) -> Set[Tuple[float, float]]:
        if self._initially_reachable_points_set is None:
            self._initially_reachable_points_set = set()
            for p in self.initially_reachable_points:
                self._initially_reachable_points_set.add(
                    self._agent_location_to_tuple(p)
                )

        return self._initially_reachable_points_set

    @property
    def currently_reachable_points(self) -> List[Dict[str, float]]:
        self.step({"action": "GetReachablePositions"})
        return self.last_event.metadata["reachablePositions"]  # type:ignore

    def get_agent_location(self) -> Dict[str, Union[float, bool]]:
        """Gets agent's location."""
        metadata = self.controller.last_event.metadata
        location = {
            "x": metadata["agent"]["position"]["x"],
            "y": metadata["agent"]["position"]["y"],
            "z": metadata["agent"]["position"]["z"],
            "rotation": metadata["agent"]["rotation"]["y"],
            "horizon": metadata["agent"]["cameraHorizon"],
            "standing": metadata["isStanding"],
        }
        return location

    def _agent_location_to_tuple(self, p):
        return (round(p["x"], 2), round(p["z"], 2))

    def get_flat_surface_grid(self, grid_size: int) -> np.ndarray:
        self.step({"action": "FlatSurfacesOnGrid", "x": grid_size, "y": grid_size})
        assert self.last_action_success
        return np.reshape(
            self.last_event.metadata["flatSurfacesOnGrid"],
            newshape=(2, grid_size, grid_size),
        )

    def get_grid_metadata(self, grid_shape: Tuple[int, int]) -> Dict[str, np.ndarray]:
        self.step(
            {"action": "GetMetadataOnGrid", "x": grid_shape[1], "y": grid_shape[0]}
        )
        assert self.last_action_success
        md = self.last_event.metadata
        return {
            "distances": np.reshape(md["distances"], grid_shape),
            "normals": np.reshape(md["normals"], (3, *grid_shape)),
            "is_openable": np.reshape(md["isOpenableGrid"], grid_shape),
        }

    @classmethod
    def allowed_offsets_for_scene(cls, map_path: str, scene_name: str, move_mag: float):
        raise NotImplementedError("Offsets disabled")

    def _snap_agent_to_initially_reachable(self, verbose=True):
        agent_location = self.get_agent_location()

        end_location_tuple = self._agent_location_to_tuple(agent_location)
        if end_location_tuple in self.initially_reachable_points_set:
            return

        agent_x = agent_location["x"]
        agent_z = agent_location["z"]

        closest_reachable_points = list(self.initially_reachable_points_set)
        closest_reachable_points = sorted(
            closest_reachable_points,
            key=lambda xz: abs(xz[0] - agent_x) + abs(xz[1] - agent_z),
        )

        # In rare cases end_location_tuple might be not considered to be in self.initially_reachable_points_set
        # even when it is, here we check for such cases.
        if (
            math.sqrt(
                (
                    (
                        np.array(closest_reachable_points[0])
                        - np.array(end_location_tuple)
                    )
                    ** 2
                ).sum()
            )
            < 1e-6
        ):
            return

        saved_last_action = self.last_action
        saved_last_action_success = self.last_action_success
        saved_last_action_return = self.last_action_return
        saved_error_message = self.last_event.metadata["errorMessage"]

        # Thor behaves weirdly when the agent gets off of the grid and you
        # try to teleport the agent back to the closest grid location. To
        # get around this we first teleport the agent to random location
        # and then back to where it should be.
        for point in self.initially_reachable_points:
            if abs(agent_x - point["x"]) > 0.1 or abs(agent_z - point["z"]) > 0.1:
                self.teleport_agent_to(
                    rotation=0,
                    horizon=30,
                    **point,
                    only_initially_reachable=False,
                    verbose=False,
                )
                if self.last_action_success:
                    break

        for p in closest_reachable_points:
            self.teleport_agent_to(
                **{**agent_location, "x": p[0], "z": p[1]},
                only_initially_reachable=False,
                verbose=False,
            )
            if self.last_action_success:
                break

        teleport_forced = False
        if not self.last_action_success:
            self.teleport_agent_to(
                **{
                    **agent_location,
                    "x": closest_reachable_points[0][0],
                    "z": closest_reachable_points[0][1],
                },
                force_action=True,
                only_initially_reachable=False,
                verbose=False,
            )
            teleport_forced = True

        self.last_action = saved_last_action
        self.last_action_success = saved_last_action_success
        self.last_action_return = saved_last_action_return
        self.last_event.metadata["errorMessage"] = saved_error_message
        new_agent_location = self.get_agent_location()
        if verbose:
            warnings.warn(
                (
                    "In {}, at location (x,z)=({},{}) which is not in the set "
                    "of initially reachable points;"
                    " attempting to correct this: agent teleported to (x,z)=({},{}).\n"
                    "Teleportation {} forced."
                ).format(
                    self.scene_name,
                    agent_x,
                    agent_z,
                    new_agent_location["x"],
                    new_agent_location["z"],
                    "was" if teleport_forced else "wasn't",
                )
            )

    def step(
        self, action_dict: Dict[str, Union[str, int, float]]
    ) -> ai2thor.server.Event:
        action = action_dict["action"]

        skip_render = "renderImage" in action_dict and not action_dict["renderImage"]
        last_frame: Optional[np.ndarray] = None
        if skip_render:
            last_frame = self.current_frame

        if self.simplify_physics:
            action_dict["simplifyOPhysics"] = True

        if "Move" in action and "Hand" not in action:  # type: ignore
            action_dict = {
                **action_dict,
                "moveMagnitude": self.move_mag,
            }  # type: ignore
            start_location = self.get_agent_location()
            sr = self.controller.step(action_dict)

            if self.restrict_to_initially_reachable_points:
                end_location_tuple = self._agent_location_to_tuple(
                    self.get_agent_location()
                )
                if end_location_tuple not in self.initially_reachable_points_set:
                    self.teleport_agent_to(**start_location, force_action=True)
                    self.last_action = action
                    self.last_action_success = False
                    self.last_event.metadata[
                        "errorMessage"
                    ] = "Moved to location outside of initially reachable points."
        elif "RandomizeHideSeekObjects" in action:
            last_position = self.get_agent_location()
            self.controller.step(action_dict)
            metadata = self.last_event.metadata
            if self.position_dist(last_position, self.get_agent_location()) > 0.001:
                self.teleport_agent_to(**last_position, force_action=True)
                warnings.warn(
                    "In scene {}, after randomization of hide and seek objects, agent moved.".format(
                        self.scene_name
                    )
                )

            sr = self.controller.step({"action": "GetReachablePositions"})
            self._initially_reachable_points = self.controller.last_event.metadata[
                "reachablePositions"
            ]
            self._initially_reachable_points_set = None
            self.last_action = action
            self.last_action_success = metadata["lastActionSuccess"]
            self.controller.last_event.metadata["reachablePositions"] = []
        elif "RotateUniverse" in action:
            sr = self.controller.step(action_dict)
            metadata = self.last_event.metadata

            if metadata["lastActionSuccess"]:
                sr = self.controller.step({"action": "GetReachablePositions"})
                self._initially_reachable_points = self.controller.last_event.metadata[
                    "reachablePositions"
                ]
                self._initially_reachable_points_set = None
                self.last_action = action
                self.last_action_success = metadata["lastActionSuccess"]
                self.controller.last_event.metadata["reachablePositions"] = []
        else:
            sr = self.controller.step(action_dict)

        if self.restrict_to_initially_reachable_points:
            self._snap_agent_to_initially_reachable()

        if skip_render:
            assert last_frame is not None
            self.last_event.frame = last_frame

        return sr

    @staticmethod
    def position_dist(
        p0: Mapping[str, Any], p1: Mapping[str, Any], ignore_y: bool = False
    ) -> float:
        return math.sqrt(
            (p0["x"] - p1["x"]) ** 2
            + (0 if ignore_y else (p0["y"] - p1["y"]) ** 2)
            + (p0["z"] - p1["z"]) ** 2
        )

    @staticmethod
    def rotation_dist(a: Dict[str, float], b: Dict[str, float]):
        def deg_dist(d0: float, d1: float):
            dist = (d0 - d1) % 360
            return min(dist, 360 - dist)

        return sum(deg_dist(a[k], b[k]) for k in ["x", "y", "z"])

    def closest_object_with_properties(
        self, properties: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        agent_pos = self.controller.last_event.metadata["agent"]["position"]
        min_dist = float("inf")
        closest = None
        for o in self.all_objects():
            satisfies_all = True
            for k, v in properties.items():
                if o[k] != v:
                    satisfies_all = False
                    break
            if satisfies_all:
                d = self.position_dist(agent_pos, o["position"])
                if d < min_dist:
                    min_dist = d
                    closest = o
        return closest

    def closest_visible_object_of_type(self, type: str) -> Optional[Dict[str, Any]]:
        properties = {"visible": True, "objectType": type}
        return self.closest_object_with_properties(properties)

    def closest_object_of_type(self, type: str) -> Optional[Dict[str, Any]]:
        properties = {"objectType": type}
        return self.closest_object_with_properties(properties)

    def closest_reachable_point_to_position(
        self, position: Dict[str, float]
    ) -> Tuple[Dict[str, float], float]:
        target = np.array([position["x"], position["z"]])
        min_dist = float("inf")
        closest_point = None
        for pt in self.initially_reachable_points:
            dist = np.linalg.norm(target - np.array([pt["x"], pt["z"]]))
            if dist < min_dist:
                closest_point = pt
                min_dist = dist
                if min_dist < 1e-3:
                    break
        assert closest_point is not None
        return closest_point, min_dist

    @staticmethod
    def _angle_from_to(a_from: float, a_to: float) -> float:
        a_from = a_from % 360
        a_to = a_to % 360
        min_rot = min(a_from, a_to)
        max_rot = max(a_from, a_to)
        rot_across_0 = (360 - max_rot) + min_rot
        rot_not_across_0 = max_rot - min_rot
        rot_err = min(rot_across_0, rot_not_across_0)
        if rot_across_0 == rot_err:
            rot_err *= -1 if a_to > a_from else 1
        else:
            rot_err *= 1 if a_to > a_from else -1
        return rot_err

    def agent_xz_to_scene_xz(self, agent_xz: Dict[str, float]) -> Dict[str, float]:
        agent_pos = self.get_agent_location()

        x_rel_agent = agent_xz["x"]
        z_rel_agent = agent_xz["z"]
        scene_x = agent_pos["x"]
        scene_z = agent_pos["z"]
        rotation = agent_pos["rotation"]
        if abs(rotation) < 1e-5:
            scene_x += x_rel_agent
            scene_z += z_rel_agent
        elif abs(rotation - 90) < 1e-5:
            scene_x += z_rel_agent
            scene_z += -x_rel_agent
        elif abs(rotation - 180) < 1e-5:
            scene_x += -x_rel_agent
            scene_z += -z_rel_agent
        elif abs(rotation - 270) < 1e-5:
            scene_x += -z_rel_agent
            scene_z += x_rel_agent
        else:
            raise Exception("Rotation must be one of 0, 90, 180, or 270.")

        return {"x": scene_x, "z": scene_z}

    def scene_xz_to_agent_xz(self, scene_xz: Dict[str, float]) -> Dict[str, float]:
        agent_pos = self.get_agent_location()
        x_err = scene_xz["x"] - agent_pos["x"]
        z_err = scene_xz["z"] - agent_pos["z"]

        rotation = agent_pos["rotation"]
        if abs(rotation) < 1e-5:
            agent_x = x_err
            agent_z = z_err
        elif abs(rotation - 90) < 1e-5:
            agent_x = -z_err
            agent_z = x_err
        elif abs(rotation - 180) < 1e-5:
            agent_x = -x_err
            agent_z = -z_err
        elif abs(rotation - 270) < 1e-5:
            agent_x = z_err
            agent_z = -x_err
        else:
            raise Exception("Rotation must be one of 0, 90, 180, or 270.")

        return {"x": agent_x, "z": agent_z}

    def all_objects(self) -> List[Dict[str, Any]]:
        return self.controller.last_event.metadata["objects"]

    def all_objects_with_properties(
        self, properties: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
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
        return self.all_objects_with_properties({"visible": True})

    def agent_relative_offset_to_absolute_offset(
        self, x_off: float, z_off: float
    ) -> Tuple[float, float]:
        rotation = self.get_agent_location()["rotation"]
        assert (
            0 <= rotation < 360 and abs(rotation % 90) < 1e-5
        ), "Rotation must be one of 0, 90, 180, or 270."

        # Theta is the amount of rotation the coordinates have undergone
        theta = -2 * np.pi * rotation / 360
        change_of_basis_mat = np.array(
            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
        )

        abs_offset = np.matmul(change_of_basis_mat, np.array([[x_off], [z_off]]))
        x_abs, z_abs = list(abs_offset[:, 0])
        return float(x_abs), float(z_abs)

    def get_object_by_id(self, object_id: str) -> Optional[Dict[str, Any]]:
        for o in self.last_event.metadata["objects"]:
            if o["objectId"] == object_id:
                return o
        return None


class AI2ThorEnvironmentMapSettings(NamedTuple):
    reachable: bool
    views: bool
    interactions: bool
    openable: bool
    cleared_fog: bool
    reveal_map_on_visit: bool = False
    reveal_map_shape: Tuple[int, int] = (4, 3)
    scene_bounds_from_reachable: bool = False
    map_padding: int = 3
    verbose: bool = True


def channels_given_environment_map_settings(settings: AI2ThorEnvironmentMapSettings):
    channels = 0
    channels += 1 if settings.reachable else 0
    channels += 4 if settings.views else 0
    channels += 12 if settings.interactions else 0
    channels += 12 if settings.openable else 0
    channels += 1 if settings.cleared_fog else 0
    return channels


class AI2ThorEnvironmentMap(object):
    def __init__(
        self,
        env: AI2ThorEnvironment,
        reveal_map_on_visit,
        reveal_map_shape: Tuple[int, int],
        scene_bounds_from_reachable: bool,
        map_padding: int = 3,
        verbose: bool = True,
    ) -> None:
        self.env = env
        self.reveal_map_on_visit = reveal_map_on_visit
        self.reveal_map_shape = reveal_map_shape
        self.verbose = verbose
        if reveal_map_shape[1] % 2 != 1 and reveal_map_shape != (0, 0):
            raise Exception(
                "reveal_map_shape must have an odd second element or equal (0,0)."
            )

        if scene_bounds_from_reachable:
            xs = [p["x"] for p in env.initially_reachable_points]
            zs = [p["z"] for p in env.initially_reachable_points]
            min_pos = {"x": min(*xs), "z": min(*zs)}
            max_pos = {"x": max(*xs), "z": max(*zs)}
            offset = env.grid_size * map_padding
        else:
            env.step({"action": "GetSceneBounds"})
            min_pos, max_pos = env.last_event.metadata["reachablePositions"]
            offset = env.grid_size * map_padding

        n = 1.0 / env.grid_size
        if abs(n - int(n)) > 0.0001:
            raise RuntimeError("Inverse of environment grid size must be an integer.")
        x_min = math.floor(n * (min_pos["x"] - offset)) / n
        x_max = math.ceil(n * (max_pos["x"] + offset)) / n
        z_min = math.floor(n * (min_pos["z"] - offset)) / n
        z_max = math.ceil(n * (max_pos["z"] + offset)) / n

        self.x_vals = list(np.linspace(x_min, x_max, round(1 + (x_max - x_min) * 4)))
        self.z_vals = list(
            reversed(np.linspace(z_min, z_max, round(1 + (z_max - z_min) * 4)))
        )
        self.x_val_to_index = {x: i for i, x in enumerate(self.x_vals)}
        self.z_val_to_index = {z: i for i, z in enumerate(self.z_vals)}

        # Reachability
        self.reachable_map_shape = (1, len(self.z_vals), len(self.x_vals))
        self.reachable_map = np.zeros(self.reachable_map_shape)
        for p in env.initially_reachable_points:
            x, z, rotation = self._clean_location(p)
            row_col = self.xz_to_row_col(x, z)
            if row_col is None:
                warnings.warn(
                    "In scene {}, position {} is initially reachable but not on the map!\nCurrent position: {}".format(
                        env.scene_name, p, env.get_agent_location()
                    )
                )
                continue
            row, col = self.xz_to_row_col(x, z)
            self.reachable_map[0, row, col] = 1.0

        # Viewed from rotation
        self.view_map_shape = (1, len(self.z_vals), len(self.x_vals))
        self.view_maps = [np.zeros(self.view_map_shape) for _ in range(4)]

        # Number of interactions with objects at height (<1m, 1m-1.5m, >1.5m) for each position/rotation
        self.interaction_map_shape = (3, len(self.z_vals), len(self.x_vals))
        self.interaction_maps = [np.zeros(self.interaction_map_shape) for _ in range(4)]

        # Number of (openable) objects open at location
        # Number of (openable) objects closed at location
        # Number of (openable) objects never opened at location (i.e. left to be explored)
        self.openable_objects_map_shape = (3, len(self.z_vals), len(self.x_vals))
        self.openable_objects_map = np.zeros(self.openable_objects_map_shape)
        openable_objects_on_map_dict = None
        if os.path.exists("resources/openable_objects_maps.json"):
            with open("resources/openable_objects_maps.json", "r") as f:
                openable_objects_on_map_dict = json.load(f)
        if (
            openable_objects_on_map_dict is None
            or self.env.scene_name not in openable_objects_on_map_dict
        ):
            if verbose:
                warnings.warn(
                    "No openable objects map data, proceeding with no object data."
                )
            openable_objects_on_map = {"map": [], "x_vals": [], "z_vals": []}
        else:
            openable_objects_on_map = openable_objects_on_map_dict[self.env.scene_name]
        (
            self.openable_object_id_to_locations,
            self.open_objects,
            self.has_been_opened,
        ) = self._initialize_openable_objects_on_map_info(
            openable_objects_on_map["map"],
            openable_objects_on_map["x_vals"],
            openable_objects_on_map["z_vals"],
        )

        # Fog
        self.fog_mask = np.zeros(self.reachable_map_shape)

        # Remember the visited locations and the agent start position
        self.visited_locations_set: Set[Tuple[int, int, int]] = set()
        self.ordered_visited_locations_list: List[Tuple[int, int, int]] = []
        self.agent_standing_list: List[bool] = []

        self.agent_start_position = self.agent_xz_to_row_col() + (
            int(self.env.get_agent_location()["rotation"]),
        )

        # Update the map
        self.update()

    def _initialize_openable_objects_on_map_info(
        self, openable_objects_on_map, x_vals, z_vals
    ):
        openable_objects = self.env.all_objects_with_properties({"openable": True})
        openable_object_ids = set(map(lambda o: o["objectId"], openable_objects))

        openable_object_ids_in_memory = set()
        object_id_to_locations = {}
        good_object_ids = set()
        for j, x in enumerate(x_vals):
            for i, z in enumerate(z_vals):
                row_col = self.xz_to_row_col(x, z)
                if row_col is not None:
                    row, col = row_col
                    for object_id in openable_objects_on_map[i][j]:
                        openable_object_ids_in_memory.add(object_id)
                        if object_id in openable_object_ids:
                            good_object_ids.add(object_id)
                            if object_id in object_id_to_locations:
                                object_id_to_locations[object_id].append((row, col))
                            else:
                                object_id_to_locations[object_id] = [(row, col)]

        should_but_dont = openable_object_ids_in_memory - openable_object_ids
        do_but_shouldnt = openable_object_ids - openable_object_ids_in_memory
        if self.verbose and len(should_but_dont) != 0:
            warnings.warn(
                "The openable objects {}, in scene {}, should be openable/exist but dont.".format(
                    should_but_dont, self.env.scene_name
                )
            )
        # if len(do_but_shouldnt) != 0:
        #     warnings.warn(
        #         "The openable objects {}, in scene {}, dont exist in the resource file.".format(
        #             do_but_shouldnt, self.env.scene_name
        #         )
        #     )

        has_been_opened = set()
        open_objects = set()
        for o in openable_objects:
            if o["isOpen"]:
                open_objects.add(o["objectId"])
                has_been_opened.add(o["objectId"])
                if o["objectId"] in good_object_ids:
                    for row, col in object_id_to_locations[o["objectId"]]:
                        self.openable_objects_map[0, row, col] += 1.0
            elif o["objectId"] in good_object_ids:
                for row, col in object_id_to_locations[o["objectId"]]:
                    self.openable_objects_map[1, row, col] += 1.0
                    self.openable_objects_map[2, row, col] += 1.0

        return object_id_to_locations, open_objects, has_been_opened

    @staticmethod
    def _ind_of_closest(needle: float, haystack: np.array) -> int:
        return int(np.argmin(np.abs(haystack - needle)))

    def agent_xz_to_row_col(self):
        agent_location = self.env.get_agent_location()
        x, z, rotation = self._clean_location(agent_location)
        agent_row_col = self.xz_to_row_col(x, z)

        if agent_row_col is not None:
            return agent_row_col
        else:
            desired_location = copy.deepcopy(agent_location)
            reachable_points = [
                (self.x_vals[col], self.z_vals[row])
                for row, col in zip(*np.where(self.reachable_map != 0)[1:])
            ]

            reachable_points = sorted(
                reachable_points, key=lambda xz: abs(xz[0] - x) + abs(xz[1] - z)
            )

            # Thor behaves weirdly when the agent gets off of the grid and you
            # try to teleport the agent back to the closest grid location. To
            # get around this we first teleport the agent to random location
            # and then back to where it should be.
            for point in self.env.initially_reachable_points:
                if abs(x - point["x"]) > 0.1 or abs(z - point["z"]) > 0.1:
                    self.env.teleport_agent_to(rotation=0, horizon=30, **point)
                    if self.env.last_action_success:
                        break

            for p in reachable_points:
                desired_location["x"] = p[0]
                desired_location["z"] = p[1]
                self.env.teleport_agent_to(**desired_location)
                if self.env.last_action_success:
                    break

            teleport_forced = False
            if not self.env.last_action_success:
                desired_location["x"] = reachable_points[0][0]
                desired_location["z"] = reachable_points[0][1]
                self.env.teleport_agent_to(**desired_location, force_action=True)
                teleport_forced = True

            new_agent_location = self.env.get_agent_location()
            warnings.warn(
                (
                    "In {}, at location (x,z)=({},{}) which is not on grid;"
                    " attempting to correct this: agent teleported to (x,z)=({},{}).\n"
                    "Teleportation {} forced."
                ).format(
                    self.env.scene_name,
                    x,
                    z,
                    new_agent_location["x"],
                    new_agent_location["z"],
                    "was" if teleport_forced else "wasn't",
                )
            )
            return self.agent_xz_to_row_col()

    def xz_to_row_col(self, x, z):
        x = round(x, 2)
        z = round(z, 2)

        z_ind = self.z_val_to_index.get(z)
        x_ind = self.x_val_to_index.get(x)

        if z_ind is None or z_ind is None:
            return None

        return z_ind, x_ind

    @staticmethod
    def _clean_location(location) -> Tuple[float, float, int]:
        return (
            round(location["x"], 2),
            round(location["z"], 2),
            0 if "rotation" not in location else round(location["rotation"]) % 360,
        )

    def _reorder_maps_given_rotation(self, view_maps, rotation):
        rotation = rotation % 360

        if rotation == 0:
            return np.concatenate(tuple(view_maps), axis=0)
        elif rotation == 90:
            return np.concatenate(tuple(view_maps[i] for i in [1, 2, 3, 0]), axis=0)
        elif rotation == 180:
            return np.concatenate(tuple(view_maps[i] for i in [2, 3, 0, 1]), axis=0)
        elif rotation == 270:
            return np.concatenate(tuple(view_maps[i] for i in [3, 0, 1, 2]), axis=0)
        else:
            raise NotImplementedError("Rotation must be in 0, 90, 180, 270")

    def update(self):
        agent_location = self.env.get_agent_location()
        _, _, rotation = self._clean_location(agent_location)
        agent_row, agent_col = self.agent_xz_to_row_col()
        self.visited_locations_set.add((agent_row, agent_col, rotation))
        self.ordered_visited_locations_list.append((agent_row, agent_col, rotation))
        self.agent_standing_list.append(agent_location["standing"])

        # Updating the reachability and fog maps
        self.fog_mask[0, agent_row, agent_col] = 1.0
        self.view_maps[rotation // 90][0, agent_row, agent_col] = 1.0
        if self.reveal_map_shape[0] != 0 and self.reveal_map_shape[1] != 0:
            if rotation == 0:
                self.fog_mask[
                    0,
                    (agent_row - self.reveal_map_shape[0]) : agent_row,
                    (agent_col - self.reveal_map_shape[1] // 2) : (
                        agent_col + 1 + self.reveal_map_shape[1] // 2
                    ),
                ] = 1.0
            elif rotation == 90:
                self.fog_mask[
                    0,
                    (agent_row - self.reveal_map_shape[1] // 2) : (
                        agent_row + 1 + self.reveal_map_shape[1] // 2
                    ),
                    (agent_col + 1) : (agent_col + 1 + self.reveal_map_shape[0]),
                ] = 1.0
            elif rotation == 180:
                self.fog_mask[
                    0,
                    (agent_row + 1) : (agent_row + 1 + self.reveal_map_shape[0]),
                    (agent_col - self.reveal_map_shape[1] // 2) : (
                        agent_col + 1 + self.reveal_map_shape[1] // 2
                    ),
                ] = 1.0
            elif rotation == 270:
                self.fog_mask[
                    0,
                    (agent_row - self.reveal_map_shape[1] // 2) : (
                        agent_row + 1 + self.reveal_map_shape[1] // 2
                    ),
                    (agent_col - self.reveal_map_shape[0]) : agent_col,
                ] = 1.0
            else:
                raise NotImplementedError("Rotation must be in 0, 90, 180, 270")

        # Updating the open object map
        for o in self.env.all_objects_with_properties({"openable": True}):
            oid = o["objectId"]
            open = o["isOpen"]
            opened = open and oid not in self.open_objects
            closed = not open and oid in self.open_objects
            changed = opened or closed
            if opened:
                self.open_objects.add(oid)
                has_been_opened_update = 1 if oid not in self.has_been_opened else 0
                self.has_been_opened.add(oid)
                if oid in self.openable_object_id_to_locations:
                    for row, col in self.openable_object_id_to_locations[oid]:
                        self.openable_objects_map[0, row, col] += 1
                        self.openable_objects_map[1, row, col] -= 1
                        self.openable_objects_map[2, row, col] -= has_been_opened_update
            elif closed:
                self.open_objects.remove(oid)
                if oid in self.openable_object_id_to_locations:
                    for row, col in self.openable_object_id_to_locations[oid]:
                        self.openable_objects_map[0, row, col] -= 1
                        self.openable_objects_map[1, row, col] += 1

            if changed:
                o_y = o["position"]["y"]
                update_ind = (o_y > 1.0) + (o_y > 1.5)
                self.interaction_maps[rotation // 90][
                    update_ind, agent_row, agent_col
                ] += 1

    def absolute_row_col_to_relative(
        self, absolute_row: int, absolute_col: int, rotation: int
    ):
        nrow = self.reachable_map_shape[1]
        ncol = self.reachable_map_shape[2]

        rotation = rotation % 360
        if rotation == 0:
            relative_row = absolute_row
            relative_col = absolute_col
        elif rotation == 90:
            relative_row = (ncol - 1) - absolute_col
            relative_col = absolute_row
        elif rotation == 180:
            relative_row = (nrow - 1) - absolute_row
            relative_col = (ncol - 1) - absolute_col
        elif rotation == 270:
            relative_row = absolute_col
            relative_col = (nrow - 1) - absolute_row
        else:
            raise ValueError("Rotation must be one of 0, 90, 180, or 270")

        return (relative_row, relative_col)

    def agent_position(self):
        absolute_row, absolute_col = self.agent_xz_to_row_col()
        x, z, rotation = self._clean_location(self.env.get_agent_location())

        nrow = self.reachable_map_shape[1]
        ncol = self.reachable_map_shape[2]

        rotation = rotation % 360
        if rotation == 0:
            ego_row = absolute_row
            ego_col = absolute_col
        elif rotation == 90:
            ego_row = (ncol - 1) - absolute_col
            ego_col = absolute_row
        elif rotation == 180:
            ego_row = (nrow - 1) - absolute_row
            ego_col = (ncol - 1) - absolute_col
        elif rotation == 270:
            ego_row = absolute_col
            ego_col = (nrow - 1) - absolute_row
        else:
            raise ValueError("Rotation must be one of 0, 90, 180, or 270")

        return {
            "x": x,
            "z": z,
            "rotation": rotation,
            "absolute_row": absolute_row,
            "absolute_col": absolute_col,
            "ego_row": ego_row,
            "ego_col": ego_col,
        }

    # NOTE: Rotation here is in the counter clockwise direction (so the new map is
    # from the perspective of an agent facing 'rotation')
    def numpy(
        self,
        rotation: int,
        reachable=True,
        views=True,
        interactions=True,
        openable=True,
        cleared_fog=True,
    ):
        if not np.any([reachable, views, interactions, openable]):
            map = np.zeros((0,) + self.reachable_map.shape)
        else:
            # This order is important!
            map = np.concatenate(
                (tuple() if not reachable else (self.reachable_map,))
                + (
                    tuple()
                    if not views
                    else (self._reorder_maps_given_rotation(self.view_maps, rotation),)
                )
                + (
                    tuple()
                    if not interactions
                    else (
                        self._reorder_maps_given_rotation(
                            self.interaction_maps, rotation
                        ),
                    )
                )
                + (tuple() if not openable else (self.openable_objects_map,))
                + (tuple() if not cleared_fog else (self.fog_mask,)),
                axis=0,
            )
            if self.reveal_map_on_visit:
                map = map * self.fog_mask

        if rotation == 0:
            pass
        elif rotation == 90:
            map = np.transpose(map, (0, 2, 1))[:, ::-1, :].copy()
        elif rotation == 180:
            map = map[:, ::-1, ::-1].copy()
        elif rotation == 270:
            map = np.transpose(map, (0, 2, 1))[:, :, ::-1].copy()
        else:
            raise ValueError("Rotation must be in 0, 90, 180, 270")

        return map
