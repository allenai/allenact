"""A wrapper for interacting with the Habitat environment"""

from typing import Dict, Union, List
import numpy as np

import habitat
from habitat.config import Config
from habitat.core.simulator import Observations, AgentState, ShortestPathPoint
from habitat.core.dataset import Episode


class HabitatEnvironment(object):
    def __init__(
        self,
        config: Config,
        docker_enabled: bool = False,
        x_display: str = None
    ) -> None:
        print("habitat env constructor")
        self.x_display = x_display
        self.env = habitat.Env(
            config=config
        )

    @property
    def scene_name(self) -> str:
        return self.env.current_episode.scene_id

    @property
    def current_frame(self) -> np.ndarray:
        return self._current_frame

    def step(
        self,
        action_dict: Dict[str, Union[str, int, float]]
    ) -> Observations:
        obs = self.env.step(action_dict["action"])
        self._current_frame = obs
        return obs

    def get_geodesic_distance(
        self,
        source_state_key: List[float],
        goal_state_key: List[float],
    ) -> float:
        return self.env.sim.geodesic_distance(source_state_key, goal_state_key)

    def get_location(self) -> AgentState:
        return self.env.sim.get_agent_state().position

    def get_shortest_path(
        self,
        source_state: AgentState,
        target_state: AgentState,
    ) -> List[ShortestPathPoint]:
        return self.env.sim.action_space_shortest_path(source_state, [target_state])

    def get_current_episode(self) -> [Episode]:
        return self.env.current_episode

    def start(self):
        print("No need to start a habitat env")

    def stop(self):
        self.env.close()

    def reset(self):
        self._current_frame = self.env.reset()

    @property
    def last_action_success(self):
        # TODO: This is a dirty hack make sure actionsa arent always successful
        return True
