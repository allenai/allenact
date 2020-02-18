from typing import Dict, Optional
import random
import copy
import warnings

import numpy as np

from rl_ai2thor.ai2thor_environment import AI2ThorEnvironment
from .robothor_constants import ROTATION_ANGLE, VISIBILITY_DISTANCE, FOV


class RoboThorEnvironment(AI2ThorEnvironment):
    def __init__(
        self,
        x_display: Optional[str] = None,
        docker_enabled: bool = False,
        local_thor_build: Optional[str] = None,
        visibility_distance: float = VISIBILITY_DISTANCE,
        fov: float = FOV,
        player_screen_width: int = 300,
        player_screen_height: int = 300,
        quality: str = "Very Low",
        restrict_to_initially_reachable_points: bool = False,
        make_agents_visible: bool = True,
        object_open_speed: float = 1.0,
        simplify_physics: bool = False,
        rotation_step_degrees: float = ROTATION_ANGLE,
    ) -> None:
        """Initializer.

        # Parameters

        x_display : The x display into which to launch ai2thor (possibly necessarily if you are running on a server
            without an attached display).
        docker_enabled : Whether or not to run thor in a docker container (useful on a server without an attached
            display so that you don't have to start an x display).
        local_thor_build : The path to a local build of ai2thor. This is probably not necessary for your use case
            and can be safely ignored.
        visibility_distance : The distance (in meters) at which objects, in the viewport of the agent,
            are considered visible by ai2thor and will have their "visible" flag be set to `True` in the metadata.
        fov : The agent's camera's field of view.
        player_screen_width : The width resolution (in pixels) of the images returned by ai2thor.
        player_screen_height : The height resolution (in pixels) of the images returned by ai2thor.
        quality : The quality at which to render. Possible quality settings can be found in
            `ai2thor._quality_settings.QUALITY_SETTINGS`.
        restrict_to_initially_reachable_points : Whether or not to restrict the agent to locations in ai2thor
            that were found to be (initially) reachable by the agent (i.e. reachable by the agent after resetting
            the scene). This can be useful if you want to ensure there are only a fixed set of locations where the
            agent can go.
        make_agents_visible : Whether or not the agent should be visible. Most noticable when there are multiple agents
            or when quality settings are high so that the agent casts a shadow.
        object_open_speed : How quickly objects should be opened. High speeds mean faster simulation but also mean
            that opening objects have a lot of kinetic energy and can, possibly, knock other objects away.
        simplify_physics : Whether or not to simplify physics when applicable. Currently this only simplies object
            interactions when opening drawers (when simplified, objects within a drawer do not slide around on
            their own when the drawer is opened or closed, instead they are effectively glued down).
        rotation_step_degrees : The angle in degrees of the agents' rotation step.
        """
        self._rotation_step_degrees = rotation_step_degrees
        super().__init__(
            x_display,
            docker_enabled,
            local_thor_build,
            visibility_distance,
            fov,
            player_screen_width,
            player_screen_height,
            quality,
            restrict_to_initially_reachable_points,
            make_agents_visible,
            object_open_speed,
            simplify_physics,
        )

    def reset(
        self, scene_name: Optional[str], move_mag: float = 0.25, **kwargs,
    ):
        """Resets the ai2thor in a new scene.

        Resets ai2thor into a new scene and initializes the scene/agents with
        prespecified settings (e.g. move magnitude).

        # Parameters

        scene_name : The scene to load.
        move_mag : The amount of distance the agent moves in a single `MoveAhead` step.
        kwargs : additional kwargs, passed to the controller "Initialize" action.
        """
        super().reset(
            scene_name,
            move_mag,
            rotateStepDegrees=self._rotation_step_degrees,
            agentMode="bot",
            fieldOfView=self._fov,
            # agentType="stochastic",
            # continuous=True,
            # snapToGrid=False,
            # renderDepthImage=True,
        )

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
        """Helper function teleporting the agent to a given location."""
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
        """Returns a random reachable location in the scene."""
        if seed is not None:
            random.seed(seed)
        xyz = random.choice(self.currently_reachable_points)
        rotation = random.choice(np.arange(0, 360, self._rotation_step_degrees))
        horizon = random.choice([0, 30, 330])
        state = copy.copy(xyz)
        state["rotation"] = rotation
        state["horizon"] = horizon
        return state

    def agent_xz_to_scene_xz(self, agent_xz: Dict[str, float]) -> Dict[str, float]:
        agent_pos = self.get_agent_location()

        x_rel_agent = agent_xz["x"]
        z_rel_agent = agent_xz["z"]
        scene_x = agent_pos["x"]
        scene_z = agent_pos["z"]
        rotation = agent_pos["rotation"]

        rads = np.pi / 180.0 * rotation
        cr, sr = np.cos(rads), np.sin(rads)

        scene_x += cr * x_rel_agent + sr * z_rel_agent
        scene_z += -sr * x_rel_agent + cr * z_rel_agent

        return {"x": scene_x, "z": scene_z}

    def scene_xz_to_agent_xz(self, scene_xz: Dict[str, float]) -> Dict[str, float]:
        agent_pos = self.get_agent_location()
        x_err = scene_xz["x"] - agent_pos["x"]
        z_err = scene_xz["z"] - agent_pos["z"]

        rotation = agent_pos["rotation"]

        rads = np.pi / 180.0 * rotation
        cr, sr = np.cos(rads), np.sin(rads)

        agent_x = cr * x_err - sr * z_err
        agent_z = sr * x_err + cr * z_err

        return {"x": agent_x, "z": agent_z}

    def reachable_points_with_rotations_and_horizons(self):
        self.controller.step({"action": "GetReachablePositions"})
        assert self.last_action_success

        points_slim = self.last_event.metadata["actionReturn"]

        points = []
        for r in np.arange(0, 360, self._rotation_step_degrees):
            for horizon in [0, 30, 330]:
                for p in points_slim:
                    p = copy.copy(p)
                    p["rotation"] = r
                    p["horizon"] = horizon
                    points.append(p)
        return points
