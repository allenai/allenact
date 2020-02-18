import typing
from typing import Any, Optional, Dict, List, Union
import logging
import random

import ai2thor
from ai2thor.controller import Controller
import numpy as np

from utils.experiment_utils import recursive_update

LOGGER = logging.getLogger("embodiedrl")


class RoboThorEnvironment:
    unity_config = dict(
        rotateStepDegrees=30,
        visibilityDistance=1.0,
        gridSize=0.25,
        agentType="stochastic",
        continuousMode=True,
        snapToGrid=False,
        agentMode="bot",
        width=640,
        height=480,
    )

    def __init__(self, **kwargs):
        recursive_update(self.unity_config, {**kwargs, "agentMode": "bot"})
        self.controller = Controller(**self.unity_config)
        self.known_good_location = self.current_state()

    def current_state(self):
        return {
            **self.last_event.metadata["agent"]["position"],
            "rotation": self.last_event.metadata["agent"]["rotation"]["y"],
            "horizon": self.last_event.metadata["agent"]["cameraHorizon"],
        }

    def reset(self, scene_name=None):
        if scene_name is not None:
            if scene_name != self.scene_name:
                self.controller.reset(scene_name)
                self.known_good_location = self.current_state()
            else:
                self.controller.step("TeleportFull", **self.known_good_location)

    def randomize_agent_location(
        self, seed: int = None, partial_position: Optional[Dict[str, float]] = None
    ) -> Dict:
        """Teleports the agent to a random reachable location in the scene."""
        if partial_position is None:
            partial_position = {}
        k = 0
        state: Optional[Dict] = None

        self.controller.step("TeleportFull", **self.known_good_location)
        while k == 0 or (not self.last_action_success and k < 10):
            state = {**self.random_reachable_state(seed=seed), **partial_position}
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

        return {
            **self.last_event.metadata["agent"]["position"],
            "rotation": self.last_event.metadata["agent"]["rotation"]["y"],
            "horizon": self.last_event.metadata["agent"]["cameraHorizon"],
        }

    def random_reachable_state(self, seed: int = None) -> Dict:
        """Returns a random reachable location in the scene."""
        if seed is not None:
            random.seed(seed)
        xyz = random.choice(self.currently_reachable_points)
        rotation = random.choice(
            np.arange(0, 360, self.unity_config["rotateStepDegrees"])
        )
        horizon = random.choice([0, 30, 330])
        return {**xyz, "rotation": rotation, "horizon": horizon}

    @property
    def currently_reachable_points(self) -> List[Dict[str, float]]:
        """List of {"x": x, "y": y, "z": z} locations in the scene that are
        currently reachable."""
        self.controller.step(action="GetReachablePositions")
        return self.last_action_return

    @property
    def scene_name(self) -> str:
        """Current ai2thor scene."""
        return self.controller.last_event.metadata["sceneName"]

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
