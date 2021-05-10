import random
from typing import Tuple, Any, List, Dict, Optional, Union, Callable, Sequence, cast

import gym
import networkx as nx
import numpy as np
from gym.utils import seeding
from gym_minigrid.envs import CrossingEnv
from gym_minigrid.minigrid import (
    DIR_TO_VEC,
    IDX_TO_OBJECT,
    MiniGridEnv,
    OBJECT_TO_IDX,
)

from allenact.base_abstractions.misc import RLStepResult
from allenact.base_abstractions.sensor import Sensor, SensorSuite
from allenact.base_abstractions.task import Task, TaskSampler
from allenact.utils.system import get_logger
from allenact_plugins.minigrid_plugin.minigrid_environments import (
    AskForHelpSimpleCrossing,
)


class MiniGridTask(Task[CrossingEnv]):
    _ACTION_NAMES: Tuple[str, ...] = ("left", "right", "forward")
    _ACTION_IND_TO_MINIGRID_IND = tuple(
        MiniGridEnv.Actions.__members__[name].value for name in _ACTION_NAMES
    )
    _CACHED_GRAPHS: Dict[str, nx.DiGraph] = {}
    """ Task around a MiniGrid Env, allows interfacing allenact with
    MiniGrid tasks. (currently focussed towards LavaCrossing)
    """

    def __init__(
        self,
        env: Union[CrossingEnv],
        sensors: Union[SensorSuite, List[Sensor]],
        task_info: Dict[str, Any],
        max_steps: int,
        task_cache_uid: Optional[str] = None,
        corrupt_expert_within_actions_of_goal: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(
            env=env, sensors=sensors, task_info=task_info, max_steps=max_steps, **kwargs
        )
        self._graph: Optional[nx.DiGraph] = None
        self._minigrid_done = False
        self._task_cache_uid = task_cache_uid
        self.corrupt_expert_within_actions_of_goal = (
            corrupt_expert_within_actions_of_goal
        )
        self.closest_agent_has_been_to_goal: Optional[float] = None

    @property
    def action_space(self) -> gym.spaces.Discrete:
        return gym.spaces.Discrete(len(self._ACTION_NAMES))

    def render(self, mode: str = "rgb", *args, **kwargs) -> np.ndarray:
        return self.env.render(mode=mode)

    def _step(self, action: int) -> RLStepResult:
        assert isinstance(action, int)
        action = cast(int, action)

        minigrid_obs, reward, self._minigrid_done, info = self.env.step(
            action=self._ACTION_IND_TO_MINIGRID_IND[action]
        )

        # self.env.render()

        return RLStepResult(
            observation=self.get_observations(minigrid_output_obs=minigrid_obs),
            reward=reward,
            done=self.is_done(),
            info=info,
        )

    def get_observations(
        self, *args, minigrid_output_obs: Optional[Dict[str, Any]] = None, **kwargs
    ) -> Any:
        return self.sensor_suite.get_observations(
            env=self.env, task=self, minigrid_output_obs=minigrid_output_obs
        )

    def reached_terminal_state(self) -> bool:
        return self._minigrid_done

    @classmethod
    def class_action_names(cls, **kwargs) -> Tuple[str, ...]:
        return cls._ACTION_NAMES

    def close(self) -> None:
        pass

    def metrics(self) -> Dict[str, Any]:
        # noinspection PyUnresolvedReferences,PyCallingNonCallable
        env_metrics = self.env.metrics() if hasattr(self.env, "metrics") else {}
        return {
            **super(MiniGridTask, self).metrics(),
            **{k: float(v) for k, v in env_metrics.items()},
            "success": int(
                self.env.was_successful
                if hasattr(self.env, "was_successful")
                else self.cumulative_reward > 0
            ),
        }

    @property
    def graph_created(self):
        return self._graph is not None

    @property
    def graph(self):
        if self._graph is None:
            if self._task_cache_uid is not None:
                if self._task_cache_uid not in self._CACHED_GRAPHS:
                    self._CACHED_GRAPHS[self._task_cache_uid] = self.generate_graph()
                self._graph = self._CACHED_GRAPHS[self._task_cache_uid]
            else:
                self._graph = self.generate_graph()
        return self._graph

    @graph.setter
    def graph(self, graph: nx.DiGraph):
        self._graph = graph

    @staticmethod
    def possible_neighbor_offsets() -> Tuple[Tuple[int, int, int], ...]:
        # Tuples of format:
        # (X translation, Y translation, rotation by 90 degrees)
        # A constant is returned, this function can be changed if anything
        # more complex needs to be done.

        # offsets_superset = itertools.product(
        #     [-1, 0, 1], [-1, 0, 1], [-1, 0, 1]
        # )
        #
        # valid_offsets = []
        # for off in offsets_superset:
        #     if (int(off[0] != 0) + int(off[1] != 0) + int(off[2] != 0)) == 1:
        #         valid_offsets.append(off)
        #
        # return tuple(valid_offsets)

        return tuple(
            [(-1, 0, 0), (0, -1, 0), (0, 0, -1), (1, 0, 0), (0, 1, 0), (0, 0, 1),]
        )

    @staticmethod
    def _add_from_to_edge(
        g: nx.DiGraph,
        s: Tuple[int, int, int],
        t: Tuple[int, int, int],
        valid_node_types: Tuple[str, ...],
    ):
        """Adds nodes and corresponding edges to existing nodes.
        This approach avoids adding the same edge multiple times.
        Pre-requisite knowledge about MiniGrid:
        DIR_TO_VEC = [
            # Pointing right (positive X)
            np.array((1, 0)),
            # Down (positive Y)
            np.array((0, 1)),
            # Pointing left (negative X)
            np.array((-1, 0)),
            # Up (negative Y)
            np.array((0, -1)),
        ]
        or
        AGENT_DIR_TO_STR = {
            0: '>',
            1: 'V',
            2: '<',
            3: '^'
        }
        This also implies turning right (clockwise) means:
            agent_dir += 1
        """

        s_x, s_y, s_rot = s
        t_x, t_y, t_rot = t

        if not (
            {g.nodes[s]["type"], g.nodes[t]["type"]}.issubset(set(valid_node_types))
        ):
            return

        x_diff = t_x - s_x
        y_diff = t_y - s_y
        angle_diff = (t_rot - s_rot) % 4

        # If source and target differ by more than one action, continue
        if sum(x != 0 for x in [x_diff, y_diff, angle_diff]) != 1 or angle_diff == 2:
            return

        xy_diff_to_agent_dir = {
            tuple(vec): dir_ind for dir_ind, vec in enumerate(DIR_TO_VEC)
        }

        action = None
        if angle_diff == 1:
            action = "right"
        elif angle_diff == 3:
            action = "left"
        elif xy_diff_to_agent_dir[(x_diff, y_diff)] == s_rot:
            # if translation is the same direction as source
            # orientation, then it's a valid forward action
            action = "forward"
        else:
            # This is when the source and target aren't one action
            # apart, despite having dx=1 or dy=1
            pass

        if action is not None:
            g.add_edge(s, t, action=action)

    def _add_node_to_graph(
        self,
        graph: nx.DiGraph,
        s: Tuple[int, int, int],
        valid_node_types: Tuple[str, ...],
        attr_dict: Dict[Any, Any] = None,
        include_rotation_free_leaves: bool = False,
    ):
        if s in graph:
            return
        if attr_dict is None:
            get_logger().warning("adding a node with neighbor checks and no attributes")
        existing_nodes = set(graph.nodes())
        graph.add_node(s, **attr_dict)

        if include_rotation_free_leaves:
            rot_free_leaf = (*s[:-1], None)
            if rot_free_leaf not in graph:
                graph.add_node(rot_free_leaf)
            graph.add_edge(s, rot_free_leaf, action="NA")

        for o in self.possible_neighbor_offsets():
            t = (s[0] + o[0], s[1] + o[1], (s[2] + o[2]) % 4)
            if t in existing_nodes:
                self._add_from_to_edge(graph, s, t, valid_node_types=valid_node_types)
                self._add_from_to_edge(graph, t, s, valid_node_types=valid_node_types)

    def generate_graph(self,) -> nx.DiGraph:
        """The generated graph is based on the fully observable grid (as the
        expert sees it all).

        env: environment to generate the graph over
        """

        image = self.env.grid.encode()
        width, height, _ = image.shape
        graph = nx.DiGraph()

        # In fully observable grid, there shouldn't be any "unseen"
        # Currently dealing with "empty", "wall", "goal", "lava"

        valid_object_ids = np.sort(
            [OBJECT_TO_IDX[o] for o in ["empty", "wall", "lava", "goal"]]
        )

        assert np.all(np.union1d(image[:, :, 0], valid_object_ids) == valid_object_ids)

        # Grid to nodes
        for x in range(width):
            for y in range(height):
                for rotation in range(4):
                    self._add_node_to_graph(
                        graph,
                        (x, y, rotation),
                        attr_dict={
                            "type": IDX_TO_OBJECT[image[x, y][0]],
                            "color": image[x, y][1],
                            "state": image[x, y][2],
                        },
                        valid_node_types=("empty", "goal"),
                    )
                    if IDX_TO_OBJECT[image[x, y][0]] == "goal":
                        if not graph.has_node("unified_goal"):
                            graph.add_node("unified_goal")
                        graph.add_edge((x, y, rotation), "unified_goal")

        return graph

    def query_expert(self, **kwargs) -> Tuple[int, bool]:
        if self._minigrid_done:
            get_logger().warning("Episode is completed, but expert is still queried.")
            return -1, False

        paths = []
        agent_x, agent_y = self.env.agent_pos
        agent_rot = self.env.agent_dir
        source_state_key = (agent_x, agent_y, agent_rot)
        assert source_state_key in self.graph

        paths.append(nx.shortest_path(self.graph, source_state_key, "unified_goal"))

        if len(paths) == 0:
            return -1, False

        shortest_path_ind = int(np.argmin([len(p) for p in paths]))

        if self.closest_agent_has_been_to_goal is None:
            self.closest_agent_has_been_to_goal = len(paths[shortest_path_ind]) - 1
        else:
            self.closest_agent_has_been_to_goal = min(
                len(paths[shortest_path_ind]) - 1, self.closest_agent_has_been_to_goal
            )

        if (
            self.corrupt_expert_within_actions_of_goal is not None
            and self.corrupt_expert_within_actions_of_goal
            >= self.closest_agent_has_been_to_goal
        ):
            return (
                int(self.env.np_random.randint(0, len(self.class_action_names()))),
                True,
            )

        if len(paths[shortest_path_ind]) == 2:
            # Since "unified_goal" is 1 step away from actual goals
            # if a path like [actual_goal, unified_goal] exists, then
            # you are already at a goal.
            get_logger().warning(
                "Shortest path computations suggest we are at"
                " the target but episode does not think so."
            )
            return -1, False

        next_key_on_shortest_path = paths[shortest_path_ind][1]
        return (
            self.class_action_names().index(
                self.graph.get_edge_data(source_state_key, next_key_on_shortest_path)[
                    "action"
                ]
            ),
            True,
        )


class ConditionedMiniGridTask(MiniGridTask):
    _ACTION_NAMES = ("left", "right", "forward", "pickup")
    _ACTION_IND_TO_MINIGRID_IND = tuple(
        MiniGridEnv.Actions.__members__[name].value for name in _ACTION_NAMES
    )

    @property
    def action_space(self) -> gym.spaces.Dict:
        return gym.spaces.Dict(
            higher=gym.spaces.Discrete(2), lower=gym.spaces.Discrete(2)
        )

    def _step(self, action: Dict[str, int]) -> RLStepResult:
        assert len(action) == 2, "got action={}".format(action)
        minigrid_obs, reward, self._minigrid_done, info = self.env.step(
            action=(
                self._ACTION_IND_TO_MINIGRID_IND[action["lower"] + 2 * action["higher"]]
            )
        )

        # self.env.render()

        return RLStepResult(
            observation=self.get_observations(minigrid_output_obs=minigrid_obs),
            reward=reward,
            done=self.is_done(),
            info=info,
        )

    def query_expert(self, **kwargs) -> Tuple[int, bool]:
        if kwargs["expert_sensor_action_group_name"] == "higher":
            if self._minigrid_done:
                get_logger().warning(
                    "Episode is completed, but expert is still queried."
                )
                return 0, False
            self.cached_expert = super().query_expert(**kwargs)
            if self.cached_expert[1]:
                return self.cached_expert[0] // 2, True
            else:
                return 0, False
        else:
            assert hasattr(self, "cached_expert")
            if self.cached_expert[1]:
                res = (self.cached_expert[0] % 2, True)
            else:
                res = (0, False)
            del self.cached_expert
            return res


class AskForHelpSimpleCrossingTask(MiniGridTask):
    _ACTION_NAMES = ("left", "right", "forward", "toggle")
    _ACTION_IND_TO_MINIGRID_IND = tuple(
        MiniGridEnv.Actions.__members__[name].value for name in _ACTION_NAMES
    )
    _CACHED_GRAPHS: Dict[str, nx.DiGraph] = {}

    def __init__(
        self,
        env: AskForHelpSimpleCrossing,
        sensors: Union[SensorSuite, List[Sensor]],
        task_info: Dict[str, Any],
        max_steps: int,
        **kwargs,
    ):
        super().__init__(
            env=env, sensors=sensors, task_info=task_info, max_steps=max_steps, **kwargs
        )

        self.did_toggle: List[bool] = []

    def _step(self, action: Union[int, Sequence[int]]) -> RLStepResult:
        assert isinstance(action, int)
        action = cast(int, action)

        self.did_toggle.append(self._ACTION_NAMES[action] == "toggle")
        return super(AskForHelpSimpleCrossingTask, self)._step(action=action)

    def metrics(self) -> Dict[str, Any]:
        return {
            **super(AskForHelpSimpleCrossingTask, self).metrics(),
            "toggle_percent": float(
                sum(self.did_toggle) / max(len(self.did_toggle), 1)
            ),
        }


class MiniGridTaskSampler(TaskSampler):
    def __init__(
        self,
        env_class: Callable[..., Union[MiniGridEnv]],
        sensors: Union[SensorSuite, List[Sensor]],
        env_info: Optional[Dict[str, Any]] = None,
        max_tasks: Optional[int] = None,
        num_unique_seeds: Optional[int] = None,
        task_seeds_list: Optional[List[int]] = None,
        deterministic_sampling: bool = False,
        cache_graphs: Optional[bool] = False,
        task_class: Callable[..., MiniGridTask] = MiniGridTask,
        repeat_failed_task_for_min_steps: int = 0,
        extra_task_kwargs: Optional[Dict] = None,
        **kwargs,
    ):
        super(MiniGridTaskSampler, self).__init__()
        self.sensors = (
            SensorSuite(sensors) if not isinstance(sensors, SensorSuite) else sensors
        )
        self.max_tasks = max_tasks
        self.num_unique_seeds = num_unique_seeds
        self.cache_graphs = cache_graphs
        self.deterministic_sampling = deterministic_sampling
        self.repeat_failed_task_for_min_steps = repeat_failed_task_for_min_steps
        self.extra_task_kwargs = (
            extra_task_kwargs if extra_task_kwargs is not None else {}
        )

        self._last_env_seed: Optional[int] = None
        self._last_task: Optional[MiniGridTask] = None
        self._number_of_steps_taken_with_task_seed = 0

        assert (not deterministic_sampling) or repeat_failed_task_for_min_steps <= 0, (
            "If `deterministic_sampling` is True then we require"
            " `repeat_failed_task_for_min_steps <= 0`"
        )
        assert (not self.cache_graphs) or self.num_unique_seeds is not None, (
            "When caching graphs you must specify"
            " a number of unique tasks to sample from."
        )
        assert (self.num_unique_seeds is None) or (
            0 < self.num_unique_seeds
        ), "`num_unique_seeds` must be a positive integer."

        self.num_unique_seeds = num_unique_seeds
        self.task_seeds_list = task_seeds_list
        if self.task_seeds_list is not None:
            if self.num_unique_seeds is not None:
                assert self.num_unique_seeds == len(
                    self.task_seeds_list
                ), "`num_unique_seeds` must equal the length of `task_seeds_list` if both specified."
            self.num_unique_seeds = len(self.task_seeds_list)
        elif self.num_unique_seeds is not None:
            self.task_seeds_list = list(range(self.num_unique_seeds))
        if num_unique_seeds is not None and repeat_failed_task_for_min_steps > 0:
            raise NotImplementedError(
                "`repeat_failed_task_for_min_steps` must be <=0 if number"
                " of unique seeds is not None."
            )

        assert (
            not self.cache_graphs
        ) or self.num_unique_seeds <= 1000, "Too many tasks (graphs) to cache"
        assert (not deterministic_sampling) or (
            self.num_unique_seeds is not None
        ), "Cannot use deterministic sampling when `num_unique_seeds` is `None`."

        if (not deterministic_sampling) and self.max_tasks:
            get_logger().warning(
                "`deterministic_sampling` is `False` but you have specified `max_tasks < inf`,"
                " this might be a mistake when running testing."
            )

        self.env = env_class(**env_info)
        self.task_class = task_class

        self.np_seeded_random_gen, _ = seeding.np_random(random.randint(0, 2 ** 31 - 1))

        self.num_tasks_generated = 0

    @property
    def length(self) -> Union[int, float]:
        return (
            float("inf")
            if self.max_tasks is None
            else self.max_tasks - self.num_tasks_generated
        )

    @property
    def total_unique(self) -> Optional[Union[int, float]]:
        return None if self.num_unique_seeds is None else self.num_unique_seeds

    @property
    def last_sampled_task(self) -> Optional[Task]:
        raise NotImplementedError

    def next_task(self, force_advance_scene: bool = False) -> Optional[MiniGridTask]:
        if self.length <= 0:
            return None

        task_cache_uid = None
        repeating = False
        if self.num_unique_seeds is not None:
            if self.deterministic_sampling:
                self._last_env_seed = self.task_seeds_list[
                    self.num_tasks_generated % len(self.task_seeds_list)
                ]
            else:
                self._last_env_seed = self.np_seeded_random_gen.choice(
                    self.task_seeds_list
                )
        else:
            if self._last_task is not None:
                self._number_of_steps_taken_with_task_seed += (
                    self._last_task.num_steps_taken()
                )

            if (
                self._last_env_seed is not None
                and self._number_of_steps_taken_with_task_seed
                < self.repeat_failed_task_for_min_steps
                and self._last_task.cumulative_reward == 0
            ):
                repeating = True
            else:
                self._number_of_steps_taken_with_task_seed = 0
                self._last_env_seed = self.np_seeded_random_gen.randint(0, 2 ** 31 - 1)

        task_has_same_seed_reset = hasattr(self.env, "same_seed_reset")

        if self.cache_graphs:
            task_cache_uid = str(self._last_env_seed)

        if repeating and task_has_same_seed_reset:
            # noinspection PyUnresolvedReferences
            self.env.same_seed_reset()
        else:
            self.env.seed(self._last_env_seed)
            self.env.saved_seed = self._last_env_seed
            self.env.reset()

        self.num_tasks_generated += 1
        task = self.task_class(
            **dict(
                env=self.env,
                sensors=self.sensors,
                task_info={},
                max_steps=self.env.max_steps,
                task_cache_uid=task_cache_uid,
            ),
            **self.extra_task_kwargs,
        )

        if repeating and self._last_task.graph_created:
            task.graph = self._last_task.graph

        self._last_task = task
        return task

    def close(self) -> None:
        self.env.close()

    @property
    def all_observation_spaces_equal(self) -> bool:
        return True

    def reset(self) -> None:
        self.num_tasks_generated = 0
        self.env.reset()

    def set_seed(self, seed: int) -> None:
        self.np_seeded_random_gen, _ = seeding.np_random(seed)
