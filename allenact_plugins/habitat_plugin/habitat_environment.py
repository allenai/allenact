"""A wrapper for interacting with the Habitat environment."""
import os
from typing import Dict, Union, List, Optional

import numpy as np

import habitat
from allenact.utils.cache_utils import DynamicDistanceCache
from allenact.utils.system import get_logger
from habitat.config import Config
from habitat.core.dataset import Dataset
from habitat.core.simulator import Observations, AgentState, ShortestPathPoint
from habitat.tasks.nav.nav import NavigationEpisode as HabitatNavigationEpisode


class HabitatEnvironment:
    def __init__(self, config: Config, dataset: Dataset, verbose: bool = False) -> None:
        self.env = habitat.Env(config=config, dataset=dataset)

        if not verbose:
            os.environ["GLOG_minloglevel"] = "2"
            os.environ["MAGNUM_LOG"] = "quiet"

        # Set the target to a random goal from the provided list for this episode
        self.goal_index = 0
        self.last_geodesic_distance = None
        self.distance_cache = DynamicDistanceCache(rounding=1)
        self._current_frame: Optional[np.ndarray] = None

    @property
    def scene_name(self) -> str:
        return self.env.current_episode.scene_id

    @property
    def current_frame(self) -> np.ndarray:
        assert self._current_frame is not None
        return self._current_frame

    def step(self, action_dict: Dict[str, Union[str, int]]) -> Observations:
        obs = self.env.step(action_dict["action"])
        self._current_frame = obs
        return obs

    def get_location(self) -> Optional[np.ndarray]:
        return self.env.sim.get_agent_state().position

    def get_rotation(self) -> Optional[List[float]]:
        return self.env.sim.get_agent_state().rotation

    def get_shortest_path(
        self,
        source_state: AgentState,
        target_state: AgentState,
    ) -> List[ShortestPathPoint]:
        return self.env.sim.action_space_shortest_path(source_state, [target_state])

    def get_current_episode(self) -> HabitatNavigationEpisode:
        return self.env.current_episode  # type: ignore

    # noinspection PyMethodMayBeStatic
    def start(self):
        get_logger().debug("No need to start a habitat_plugin env")

    def stop(self):
        self.env.close()

    def reset(self):
        self._current_frame = self.env.reset()

    @property
    def last_action_success(self) -> bool:
        # For now we can not have failure of actions
        return True

    @property
    def num_episodes(self) -> int:
        ep_iterator = self.env.episode_iterator
        assert isinstance(ep_iterator, habitat.core.dataset.EpisodeIterator)
        return len(ep_iterator.episodes)
