"""A wrapper for interacting with the Habitat environment"""

from typing import Dict, Union, List
import numpy as np
import random

import habitat
from habitat.config import Config
from habitat.core.simulator import Observations, AgentState, ShortestPathPoint
from habitat.core.dataset import Episode, Dataset


class HabitatEnvironment(object):
    def __init__(
        self,
        config: Config,
        dataset: Dataset,
        x_display: str = None
    ) -> None:
        print("rl_habitat env constructor")
        self.x_display = x_display
        self.env = habitat.Env(
            config=config,
            dataset=dataset
        )
        # Set the target to a random goal from the provided list for this episode
        self.goal_index = 0
        self.last_geodesic_distance = None

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

    def get_geodesic_distance(self) -> float:
        curr = self.get_location()
        goals = self.get_current_episode().goals

        goal = goals[self.goal_index].position
        distance = self.env.sim.geodesic_distance(curr, goal)

        # If the distance to the current goal can not be computed, then attempt
        # to compute to distance to any goal in the episode and keep that as the
        # new goal for the duration of the episode (or until the distance to this
        # goal can no longer be computed)
        try_count = 0
        while distance in [float('-inf'), float('inf')] or np.isnan(distance):
            self.goal_index = (self.goal_index + 1) % len(goals)
            goal = goals[self.goal_index].position
            distance = self.env.sim.geodesic_distance(curr, goal)
            try_count += 1
            if try_count >= len(goals):
                print("In here returning 0")
                if self.last_geodesic_distance is None:
                    self.last_geodesic_distance = 0.0
                    return 0.0
                return self.last_geodesic_distance
        print("Distance:", distance)
        self.last_geodesic_distance = distance
        return distance

    def get_distance_to_target(self) -> float:
        curr = self.get_location()
        goal = self.get_current_episode().goals[0].view_points[0].agent_state.position
        return self.env.sim.geodesic_distance(curr, goal)

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
        print("No need to start a rl_habitat env")

    def stop(self):
        self.env.close()

    def reset(self):
        self._current_frame = self.env.reset()
        # Set the target to a random goal from the provided list for this episode
        self.goal_index = random.randint(0, len(self.get_current_episode().goals))
        # Reset the last distance to target
        self.get_geodesic_distance()

    @property
    def last_action_success(self) -> bool:
        # We can not pick things up so we really can't fail in our actions
        return True

    @property
    def num_episodes(self) -> int:
        return len(self.env.episode_iterator.episodes)
