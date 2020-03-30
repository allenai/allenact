import copy
import curses
import itertools
import random
import time
from functools import lru_cache
from typing import Optional, Tuple, Any, List, Union

import numpy as np
import typing

EMPTY = 0
GOAL = 1
WRONG_CORNER = 2
WALL = 3


@lru_cache(1000)
def _get_world_corners(world_dim: int, world_radius: int):
    if world_radius == 0:
        return ((0,) * world_dim,)

    def combination_to_vec(comb) -> Tuple[int, ...]:
        vec = [world_radius] * world_dim
        for k in comb:
            vec[k] *= -1
        return tuple(vec)

    return tuple(
        combination_to_vec(comb)
        for i in range(world_dim + 1)
        for comb in itertools.combinations(list(range(world_dim)), i)
    )


@lru_cache(1000)
def _base_world_tensor(world_dim: int, world_radius: int):
    tensor = np.full((2 * world_radius + 1,) * world_dim, fill_value=EMPTY)

    slices: List[Union[slice, int]] = [slice(0, 2 * world_radius + 1)] * world_dim
    for i in range(world_dim):
        tmp_slices = [*slices]
        tmp_slices[i] = 0
        tensor[tuple(tmp_slices)] = WALL
        tmp_slices[i] = 2 * world_radius
        tensor[tuple(tmp_slices)] = WALL

    for corner in _get_world_corners(world_dim=world_dim, world_radius=world_radius):
        tensor[tuple([loc + world_radius for loc in corner])] = WRONG_CORNER

    return tensor


class LightHouseEnvironment(object):
    EMPTY = 0
    GOAL = 1
    WRONG_CORNER = 2
    WALL = 3
    SPACE_LEVELS = [EMPTY, GOAL, WRONG_CORNER, WALL]

    def __init__(self, world_dim: int, world_radius: int, **kwargs):
        self.world_dim = world_dim
        self.world_radius = world_radius

        self.world_corners = np.array(
            _get_world_corners(world_dim=world_dim, world_radius=world_radius),
            dtype=int,
        )

        self.curses_screen: Optional[Any] = None

        self.world_tensor: np.ndarray = copy.deepcopy(
            _base_world_tensor(world_radius=world_radius, world_dim=world_dim)
        )
        self.current_position = np.zeros(world_dim, dtype=int)
        self.closest_distance_to_corners = np.full(
            2 ** world_dim, fill_value=world_radius, dtype=int
        )
        self.positions: List[Tuple[int, ...]] = [tuple(self.current_position)]
        self.goal_position: Optional[np.ndarray] = None
        self.last_action: Optional[int] = None

        self.random_reset()

    def random_reset(self, goal_position: Optional[bool] = None):
        self.last_action = None
        self.world_tensor = copy.deepcopy(
            _base_world_tensor(world_radius=self.world_radius, world_dim=self.world_dim)
        )
        if goal_position is None:
            self.goal_position = random.choice(self.world_corners)
        self.world_tensor[
            tuple(typing.cast(np.ndarray, self.world_radius + self.goal_position))
        ] = GOAL

        if self.curses_screen is not None:
            curses.nocbreak()
            self.curses_screen.keypad(False)
            curses.echo()
            curses.endwin()

        self.curses_screen = None

        self.current_position = np.zeros(self.world_dim, dtype=int)
        self.closest_distance_to_corners = np.abs(
            (self.world_corners - self.current_position.reshape(1, -1))
        ).max(1)

        self.positions = [tuple(self.current_position)]

    # @classmethod
    # def state_space_size(cls, world_dim) -> int:
    #     return (3 if world_dim == 1 else 4) ** (2 ** world_dim)

    def step(self, action: int) -> bool:
        assert 0 <= action < 2 * self.world_dim
        self.last_action = action

        delta = -1 if action >= self.world_dim else 1
        ind = action % self.world_dim
        old = self.current_position[ind]
        new = min(max(delta + old, -self.world_radius), self.world_radius)
        if new == old:
            self.positions.append(self.positions[-1])
            return False
        else:
            self.current_position[ind] = new
            self.closest_distance_to_corners = np.minimum(
                np.abs((self.world_corners - self.current_position.reshape(1, -1))).max(
                    1
                ),
                self.closest_distance_to_corners,
            )
            self.positions.append(tuple(self.current_position))
            return True

    def render(self, mode="array", **kwargs):
        if mode == "array":
            arr = copy.deepcopy(self.world_tensor)
            arr[tuple(self.world_radius + self.current_position)] = 9
            return arr

        assert mode == "curses"

        if self.world_dim == 1:
            space_list = ["_"] * (1 + 2 * self.world_radius)

            goal_ind = self.goal_position[0] + self.world_radius
            space_list[goal_ind] = "G"
            space_list[2 * self.world_radius - goal_ind] = "W"
            space_list[self.current_position[0] + self.world_radius] = "X"

            to_print = " ".join(space_list)

            if self.curses_screen is None:
                self.curses_screen = curses.initscr()

            self.curses_screen.addstr(0, 0, to_print)
            if "extra_text" in kwargs:
                self.curses_screen.addstr(1, 0, kwargs["extra_text"])
            self.curses_screen.refresh()
        elif self.world_dim == 2:
            space_list = [
                ["_"] * (1 + 2 * self.world_radius)
                for _ in range(1 + 2 * self.world_radius)
            ]

            for row_ind in range(1 + 2 * self.world_radius):
                for col_ind in range(1 + 2 * self.world_radius):
                    if self.world_tensor[row_ind][col_ind] == self.GOAL:
                        space_list[row_ind][col_ind] = "G"

                    if self.world_tensor[row_ind][col_ind] == self.WRONG_CORNER:
                        space_list[row_ind][col_ind] = "C"

                    if self.world_tensor[row_ind][col_ind] == self.WALL:
                        space_list[row_ind][col_ind] = "W"

                    if (
                        (row_ind, col_ind) == self.world_radius + self.current_position
                    ).all():
                        space_list[row_ind][col_ind] = "X"

            if self.curses_screen is None:
                self.curses_screen = curses.initscr()

            for i, sl in enumerate(space_list):
                self.curses_screen.addstr(i, 0, " ".join(sl))

            self.curses_screen.addstr(len(space_list), 0, str(self.state()))
            if "extra_text" in kwargs:
                self.curses_screen.addstr(len(space_list) + 1, 0, kwargs["extra_text"])

            self.curses_screen.refresh()
        else:
            raise NotImplementedError("Cannot render worlds of > 2 dimensions.")

        time.sleep(0.0 if "sleep_time" not in kwargs else kwargs["sleep_time"])

    def close(self):
        if self.curses_screen is not None:
            curses.nocbreak()
            self.curses_screen.keypad(False)
            curses.echo()
            curses.endwin()

    # @staticmethod
    # def get_probability_of_states_for_expert_policy(
    #     env: "LightHouseEnvironment",
    #     expert_view_level: int,
    #     additional_view_levels_list: Optional[List[int]] = None,
    # ) -> Tuple[Dict[HashableDict, int], Dict]:
    #     assert all(vl <= expert_view_level for vl in additional_view_levels_list)
    #
    #     def _helper(
    #         env: "LightHouseEnvironment",
    #         current_prob: float,
    #         expert_state_to_prob: Dict[defaultdict, float],
    #         view_level_to_state_to_target_dist: Dict[int, Dict[defaultdict, Any]],
    #         ep_length: int = 0,
    #     ):
    #         expert_state = env.state()
    #         action_probs = env.true_expert_policy(**expert_state).view(-1)
    #
    #         new_path_lens_with_probs = []
    #         for i, p in enumerate(action_probs.numpy().tolist()):
    #             if p > 0:
    #                 tmpenv = env.clone()
    #                 _, _, done, _ = tmpenv.step(i)
    #                 if not done:
    #                     path_lens_with_probs: List[Tuple[int, float]] = _helper(
    #                         tmpenv,
    #                         current_prob * p,
    #                         expert_state_to_prob,
    #                         view_level_to_state_to_target_dist,
    #                         ep_length=ep_length + 1,
    #                     )
    #                     new_path_lens_with_probs.extend(
    #                         (l, path_prob) for l, path_prob in path_lens_with_probs
    #                     )
    #                 else:
    #                     new_path_lens_with_probs.append(
    #                         (ep_length + 1, current_prob * p)
    #                     )
    #
    #         weight = sum(p / l for (l, p) in new_path_lens_with_probs)
    #         expert_state_to_prob[expert_state] += weight
    #
    #         for view_level in view_level_to_state_to_target_dist:
    #             state_to_target_dist = view_level_to_state_to_target_dist[view_level]
    #             t = env.state(view_level=view_level)
    #
    #             if t not in state_to_target_dist:
    #                 state_to_target_dist[t] = [torch.zeros(env.action_space.n), 0]
    #
    #             state_to_target_dist[t][0] += weight * action_probs
    #             state_to_target_dist[t][1] += weight
    #
    #         return new_path_lens_with_probs
    #
    #     expert_state_to_prob = defaultdict(lambda: 0)
    #     view_level_to_state_to_target_dist = {
    #         k: {} for k in additional_view_levels_list
    #     }
    #
    #     expert_env = env.clone(new_view_level=expert_view_level)
    #
    #     goal_positions = env.goal_positions()
    #     for goal_position in goal_positions:
    #         expert_env.reset(goal_position=goal_position)
    #         _helper(
    #             expert_env,
    #             1 / len(goal_positions),
    #             expert_state_to_prob=expert_state_to_prob,
    #             view_level_to_state_to_target_dist=view_level_to_state_to_target_dist,
    #         )
    #
    #     return expert_state_to_prob, view_level_to_state_to_target_dist


# class OneDimEnvironment(HExEnv):
#     metadata = {"render.modes": ["human"]}
#
#     EMPTY = 0
#     GOAL = 1
#     WRONG_SIDE = 2
#     SPACE_LEVELS = [EMPTY, GOAL, WRONG_SIDE]
#
#     LEFT_ACTION = 0
#     RIGHT_ACTION = 1
#     NONE_ACTION = 2
#     LAST_ACTION_LEVELS = [LEFT_ACTION, RIGHT_ACTION, NONE_ACTION]
#
#     STEP_PENALTY = -0.01
#     SUCCESS_REWARD = 1.0
#
#     ACTION_MEMORY_SIZE = 1
#
#     action_space = gym.spaces.Discrete(2)
#
#     __tuple_to_ind = None
#     __design_matrix = None
#
#     @classmethod
#     def _create_formula(cls):
#         def f(input):
#             return "C({}, levels=OneDimEnvironment.SPACE_LEVELS)".format(input)
#
#         def g(input):
#             return "C({}, levels=OneDimEnvironment.LAST_ACTION_LEVELS)".format(input)
#
#         return ":".join(
#             [f("s0"), f("s1"), f("s2")]
#             + [g("a{}".format(i)) for i in range(cls.ACTION_MEMORY_SIZE)]
#         )
#
#     @classmethod
#     def _tuple_to_ind(cls, t: Tuple):
#         if cls.__tuple_to_ind is None:
#             cls._design_matrix()
#         return cls.__tuple_to_ind[t]
#
#     @classmethod
#     def _design_matrix(cls):
#         if cls.__design_matrix is None:
#             cls.__design_matrix = cls._create_full_design_matrix()
#         return cls.__design_matrix
#
#     def __init__(
#         self,
#         world_radius: int = 1,
#         visibility_radius: Optional[int] = None,
#         view_level: Optional[int] = None,
#         remember_seen_states: bool = False,
#         **kwargs
#     ):
#         assert visibility_radius is not None or view_level is not None
#         assert visibility_radius is None or view_level is None
#
#         if visibility_radius is None:
#             visibility_radius = self.view_level_to_visibility_radius(view_level)
#
#         assert world_radius > 0 and visibility_radius >= 0
#
#         self.world_radius = world_radius
#         self.visibility_radius = visibility_radius
#         self.remember_seen_states = remember_seen_states
#
#         self.curses_screen = None
#
#         self.reset()
#
#     def reset(self, goal_position: Optional[bool] = None):
#         self.goal_on_right = (
#             goal_position if goal_position is not None else random.random() < 0.5
#         )
#         self.goal_position_x = self.world_radius * (1 if self.goal_on_right else -1)
#         self.wrong_side_position = -1 * self.goal_position_x
#         if self.curses_screen is not None:
#             curses.nocbreak()
#             self.curses_screen.keypad(False)
#             curses.echo()
#             curses.endwin()
#
#         self.curses_screen = None
#
#         self._position_to_type_offset = np.array(
#             [self.GOAL if not self.goal_on_right else self.WRONG_SIDE]
#             + [self.EMPTY] * (1 + (self.world_radius - 1) * 2)
#             + [self.GOAL if self.goal_on_right else self.WRONG_SIDE]
#         )
#
#         self.current_position = 0
#         self.furthest_left = 0
#         self.furthest_right = 0
#         self.positions = [0]
#         self.last_actions_queue = Queue()
#         for _ in range(self.ACTION_MEMORY_SIZE):
#             self.last_actions_queue.put(self.NONE_ACTION)
#
#         return self.state()
#
#     def clone(self, new_view_level: Optional[int] = None):
#         cloned = OneDimEnvironment(
#             world_radius=self.world_radius,
#             view_level=new_view_level
#             if new_view_level is not None
#             else self.current_view_level,
#             remember_seen_states=self.remember_seen_states,
#         )
#         cloned.goal_on_right = self.goal_on_right
#         cloned.goal_position_x = self.goal_position_x
#         cloned.wrong_side_position = self.wrong_side_position
#         cloned._position_to_type_offset = self._position_to_type_offset
#         cloned.current_position = self.current_position
#         cloned.furthest_left = self.furthest_left
#         cloned.furthest_right = self.furthest_right
#         cloned.positions = copy.deepcopy(self.positions)
#
#         cloned.last_actions_queue = Queue()
#         for x in list(self.last_actions_queue.queue):
#             cloned.last_actions_queue.put(x)
#
#         return cloned
#
#     def goal_positions(self) -> List[bool]:
#         return [False, True]
#
#     @property
#     def num_view_levels(self) -> int:
#         return self.world_radius * 2 + 2
#
#     @property
#     def current_view_level(self) -> int:
#         return self.visibility_radius_to_view_level(
#             visibility_radius=self.visibility_radius
#         )
#
#     def visibility_radius_to_view_level(self, visibility_radius: int):
#         return visibility_radius
#
#     def view_level_to_visibility_radius(self, view_level: int):
#         return view_level
#
#     @classmethod
#     def state_space_size(cls) -> int:
#         return (len(cls.SPACE_LEVELS) ** 3) * (
#             len(cls.LAST_ACTION_LEVELS)
#         ) ** cls.ACTION_MEMORY_SIZE
#
#     def step(self, action):
#         if action == self.LEFT_ACTION:
#             self.current_position = max(self.current_position - 1, -self.world_radius)
#             self.furthest_left = min(self.furthest_left, self.current_position)
#         elif action == self.RIGHT_ACTION:
#             self.current_position = min(self.current_position + 1, self.world_radius)
#             self.furthest_right = max(self.furthest_right, self.current_position)
#         else:
#             raise NotImplementedError()
#         self.positions.append(self.current_position)
#
#         self.last_actions_queue.get()
#         self.last_actions_queue.put(action)
#
#         # take action and get reward, transit to next state
#         done = self.current_position == self.goal_position_x
#         return (
#             self.state(),
#             self.SUCCESS_REWARD if done else self.STEP_PENALTY,
#             done,
#             {},
#         )
#
#     def render(self, mode="human", sleep_time=0.01, **kwargs):
#         space_list = ["_"] * (1 + 2 * self.world_radius)
#
#         space_list[self.goal_position_x + self.world_radius] = "G"
#         space_list[-self.goal_position_x + self.world_radius] = "W"
#         space_list[self.current_position + self.world_radius] = "X"
#
#         to_print = " ".join(space_list)
#
#         if self.curses_screen is None:
#             self.curses_screen = curses.initscr()
#
#         self.curses_screen.addstr(0, 0, to_print)
#         if "extra_text" in kwargs:
#             self.curses_screen.addstr(1, 0, kwargs["extra_text"])
#         self.curses_screen.refresh()
#         time.sleep(sleep_time)
#
#         # print(" ".join(space_list), end='\r', flush=True)
#
#     def view_tuple_given_position_and_visibility(
#         self, position: int, visibility_radius: int
#     ):
#         if self.remember_seen_states:
#             assert self.furthest_left <= position <= self.furthest_right
#             t = (
#                 position,
#                 max(self.furthest_left - visibility_radius, -self.world_radius),
#                 min(self.furthest_right + visibility_radius, self.world_radius),
#             )
#         else:
#             t = (
#                 position,
#                 max(position - visibility_radius, -self.world_radius),
#                 min(position + visibility_radius, self.world_radius),
#             )
#
#         return tuple(self._position_to_type_offset[tt + self.world_radius] for tt in t)
#
#     def view_tuple_for_visibility_radius(self, visibility_radius: int):
#         return self.view_tuple_given_position_and_visibility(
#             position=self.current_position, visibility_radius=visibility_radius
#         )
#
#     def last_action(self):
#         return list(self.last_actions_queue.queue)[-1]
#
#     def state(self, view_level: Optional[int] = None) -> HashableDict:
#         if view_level is None:
#             view_level = self.current_view_level
#
#         visibility_radius = self.view_level_to_visibility_radius(view_level)
#         return HashableDict(
#             view_tuple=self.view_tuple_for_visibility_radius(visibility_radius),
#             last_action=self.last_action(),
#         )
#
#     def prepare_state_for_model(
#         self,
#         view_tuple: Tuple[int, int, int],
#         last_action: int,
#         view_level: Optional[int] = None,
#     ):
#         return {
#             "x": (
#                 torch.from_numpy(
#                     self._design_matrix()[
#                         self._tuple_to_ind(view_tuple + (last_action,)), :
#                     ]
#                 )
#                 .float()
#                 .unsqueeze(0)
#             ),
#             "view_tuple": view_tuple,
#             "last_action": last_action,
#         }
#
#     def close(self):
#         pass
#
#     @classmethod
#     def true_expert_policy(cls, view_tuple, last_action, **kwargs):
#         cur_view, left_view, right_view = view_tuple
#
#         goal = cls.GOAL
#         wrong_side = cls.WRONG_SIDE
#         left = cls.LEFT_ACTION
#         right = cls.RIGHT_ACTION
#
#         if left_view == goal:
#             expert_action = left
#         elif right_view == goal:
#             expert_action = right
#         elif cur_view == wrong_side:
#             expert_action = left if last_action == right else right
#         elif left_view == wrong_side:
#             expert_action = right
#         elif right_view == wrong_side:
#             expert_action = left
#         elif last_action == cls.NONE_ACTION:
#             return torch.FloatTensor([[0.5, 0.5]])
#         else:
#             expert_action = last_action
#
#         return torch.FloatTensor([[expert_action == left, expert_action == right]])
#
#     def optimal_average_steps(self, *args, **kwargs):
#         world_radius = (
#             self.world_radius
#             if "world_radius" not in kwargs
#             else kwargs["world_radius"]
#         )
#         visibility_radius = (
#             self.visibility_radius
#             if "visibility_radius" not in kwargs
#             else kwargs["visibility_radius"]
#         )
#
#         max_steps_wrong_dir = max(world_radius - visibility_radius, 0)
#
#         return max_steps_wrong_dir + world_radius
#
#     def optimal_average_reward(self, *args, **kwargs):
#         return self.SUCCESS_REWARD + self.STEP_PENALTY * (
#             self.optimal_average_steps() - 1
#         )
#
#
# class TwoDimEnvironment(HExEnv):
#     metadata = {"render.modes": ["human"]}
#
#     EMPTY = 0
#     GOAL = 1
#     WRONG_CORNER = 2
#     WALL = 3
#     SPACE_LEVELS = [EMPTY, GOAL, WRONG_CORNER, WALL]
#
#     UP_ACTION = 0
#     RIGHT_ACTION = 1
#     DOWN_ACTION = 2
#     LEFT_ACTION = 3
#     NONE_ACTION = 4
#     LAST_ACTION_LEVELS = [
#         UP_ACTION,
#         RIGHT_ACTION,
#         DOWN_ACTION,
#         LEFT_ACTION,
#         NONE_ACTION,
#     ]
#
#     TOP_RIGHT = 0
#     BOTTOM_RIGHT = 1
#     BOTTOM_LEFT = 2
#     TOP_LEFT = 3
#     GOAL_POSITIONS = [TOP_RIGHT, BOTTOM_LEFT, BOTTOM_RIGHT, TOP_LEFT]
#
#     STEP_PENALTY = -0.01
#     SUCCESS_REWARD = 1.0
#
#     ACTION_MEMORY_SIZE = 1
#
#     action_space = gym.spaces.Discrete(4)
#
#     def __init__(
#         self,
#         world_radius: int = 1,
#         visibility_radius: Optional[int] = None,
#         view_level: Optional[int] = None,
#         remember_seen_states: bool = False,
#         **kwargs
#     ):
#         assert visibility_radius is not None or view_level is not None
#         assert visibility_radius is None or view_level is None
#
#         if visibility_radius is None:
#             visibility_radius = self.view_level_to_visibility_radius(view_level)
#         assert world_radius > 0 and visibility_radius >= 0
#
#         self.world_radius = world_radius
#         self.visibility_radius = visibility_radius
#         self.remember_seen_states = remember_seen_states
#
#         self.tuple_to_ind = {}
#         self.design_matrix = torch.from_numpy(self._create_full_design_matrix()).float()
#         self.curses_screen = None
#
#         self.reset()
#
#     def _create_formula(self):
#         def f(input):
#             return "C({}, levels=TwoDimEnvironment.SPACE_LEVELS)".format(input)
#
#         def g(input):
#             return "C({}, levels=TwoDimEnvironment.LAST_ACTION_LEVELS)".format(input)
#
#         return ":".join(
#             [g("hitting")]
#             + [f("s{}".format(i)) for i in range(4)]
#             + [g("a{}".format(i)) for i in range(self.ACTION_MEMORY_SIZE)]
#         )
#
#     @property
#     def num_view_levels(self) -> int:
#         return self.world_radius * 2 + 2
#
#     @property
#     def current_view_level(self) -> int:
#         return self.visibility_radius_to_view_level(
#             visibility_radius=self.visibility_radius
#         )
#
#     def visibility_radius_to_view_level(self, visibility_radius: int):
#         return visibility_radius
#
#     def view_level_to_visibility_radius(self, view_level: int):
#         return view_level
#
#     @classmethod
#     def state_space_size(cls) -> int:
#         return (
#             len(cls.LAST_ACTION_LEVELS)
#             * (len(cls.SPACE_LEVELS) ** 4)
#             * (len(cls.LAST_ACTION_LEVELS)) ** cls.ACTION_MEMORY_SIZE
#         )
#
#     def _goal_position_to_xy(self, goal_position: int):
#         wr = self.world_radius
#         if goal_position == self.TOP_RIGHT:
#             xy = (wr, wr)
#         elif goal_position == self.BOTTOM_RIGHT:
#             xy = (wr, -wr)
#         elif goal_position == self.BOTTOM_LEFT:
#             xy = (-wr, -wr)
#         elif goal_position == self.TOP_LEFT:
#             xy = (-wr, wr)
#         else:
#             raise NotImplementedError()
#         return xy
#
#     def reset(self, goal_position: int = None):
#         if goal_position is not None:
#             self.current_goal_position = goal_position
#         else:
#             self.current_goal_position = random.choice(self.GOAL_POSITIONS)
#         self.current_goal_position_as_xy = self._goal_position_to_xy(
#             self.current_goal_position
#         )
#         if self.curses_screen is not None:
#             curses.nocbreak()
#             self.curses_screen.keypad(False)
#             curses.echo()
#             curses.endwin()
#
#         self.curses_screen = None
#
#         self.current_x = 0
#         self.current_y = 0
#         self.positions = [(0, 0)]
#         self.last_actions_queue = Queue()
#         for _ in range(self.ACTION_MEMORY_SIZE):
#             self.last_actions_queue.put(self.NONE_ACTION)
#
#         return self.state()
#
#     def _position_to_type_offset(self, x, y):
#         x_at_edge = abs(x) == self.world_radius
#         y_at_edge = abs(y) == self.world_radius
#         if x_at_edge or y_at_edge:
#             if x_at_edge and y_at_edge:
#                 if (x, y) == self.current_goal_position_as_xy:
#                     return self.GOAL
#                 else:
#                     return self.WRONG_CORNER
#             else:
#                 return self.WALL
#         else:
#             return self.EMPTY
#
#     def step(self, action):
#         if action == self.UP_ACTION:
#             self.current_y = min(self.current_y + 1, self.world_radius)
#         elif action == self.DOWN_ACTION:
#             self.current_y = max(self.current_y - 1, -self.world_radius)
#         elif action == self.RIGHT_ACTION:
#             self.current_x = min(self.current_x + 1, self.world_radius)
#         elif action == self.LEFT_ACTION:
#             self.current_x = max(self.current_x - 1, -self.world_radius)
#         else:
#             raise NotImplementedError()
#         self.positions.append((self.current_x, self.current_y))
#
#         self.last_actions_queue.get()
#         self.last_actions_queue.put(action)
#
#         # take action and get reward, transit to next state
#         done = self.positions[-1] == self.current_goal_position_as_xy
#         return (
#             self.state(),
#             self.SUCCESS_REWARD if done else self.STEP_PENALTY,
#             done,
#             {},
#         )
#
#     def view_tuple_given_position_and_visibility(
#         self, xpos: int, ypos: int, visibility_radius: int
#     ):
#         def clamp(v):
#             return min(max(v, -self.world_radius), self.world_radius)
#
#         tmp = [(1, 1), (1, -1), (-1, -1), (-1, 1)]
#
#         hitting = self.NONE_ACTION
#         if xpos == self.world_radius:
#             hitting = self.RIGHT_ACTION
#         elif xpos == -self.world_radius:
#             hitting = self.LEFT_ACTION
#         elif ypos == self.world_radius:
#             hitting = self.UP_ACTION
#         elif ypos == -self.world_radius:
#             hitting = self.DOWN_ACTION
#
#         return (hitting,) + tuple(
#             self._position_to_type_offset(
#                 x=clamp(xpos + i * visibility_radius),
#                 y=clamp(ypos + j * visibility_radius),
#             )
#             for i, j in tmp
#         )
#
#     def view_tuple_for_visibility_radius(self, visibility_radius: int):
#         return self.view_tuple_given_position_and_visibility(
#             xpos=self.current_x,
#             ypos=self.current_y,
#             visibility_radius=visibility_radius,
#         )
#
#     def last_action(self):
#         return list(self.last_actions_queue.queue)[-1]
#
#     def state(self, view_level: Optional[int] = None) -> HashableDict:
#         if view_level is None:
#             view_level = self.current_view_level
#
#         visibility_radius = self.view_level_to_visibility_radius(view_level)
#         return HashableDict(
#             view_tuple=self.view_tuple_for_visibility_radius(visibility_radius),
#             last_action=self.last_action(),
#         )
#
#     def prepare_state_for_model(
#         self,
#         view_tuple: Tuple[int, int, int, int, int],
#         last_action: int,
#         view_level: Optional[int] = None,
#     ):
#         return {
#             "x": (
#                 self.design_matrix[
#                     self.tuple_to_ind[view_tuple + (last_action,)], :
#                 ].unsqueeze(0)
#             ),
#             "view_tuple": view_tuple,
#             "last_action": last_action,
#         }
#
#     def close(self):
#         pass
#
#     @classmethod
#     def true_expert_policy(cls, view_tuple, last_action, **kwargs):
#         hitting, tr, br, bl, tl = view_tuple
#
#         goal = cls.GOAL
#         wrong = cls.WRONG_CORNER
#         wall = cls.WALL
#
#         u, r, d, l = cls.UP_ACTION, cls.RIGHT_ACTION, cls.DOWN_ACTION, cls.LEFT_ACTION
#
#         if tr == goal:
#             if hitting != r:
#                 expert_action = r
#             else:
#                 expert_action = u
#         elif br == goal:
#             if hitting != d:
#                 expert_action = d
#             else:
#                 expert_action = r
#         elif bl == goal:
#             if hitting != l:
#                 expert_action = l
#             else:
#                 expert_action = d
#         elif tl == goal:
#             if hitting != u:
#                 expert_action = u
#             else:
#                 expert_action = l
#
#         elif tr == wrong and not any(x == wrong for x in [br, bl, tl]):
#             expert_action = l
#         elif br == wrong and not any(x == wrong for x in [bl, tl, tr]):
#             expert_action = u
#         elif bl == wrong and not any(x == wrong for x in [tl, tr, br]):
#             expert_action = r
#         elif tl == wrong and not any(x == wrong for x in [tr, br, bl]):
#             expert_action = d
#
#         elif all(x == wrong for x in [tr, br]) and not any(
#             x == wrong for x in [bl, tl]
#         ):
#             expert_action = l
#         elif all(x == wrong for x in [br, bl]) and not any(
#             x == wrong for x in [tl, tr]
#         ):
#             expert_action = u
#
#         elif all(x == wrong for x in [bl, tl]) and not any(
#             x == wrong for x in [tr, br]
#         ):
#             expert_action = r
#         elif all(x == wrong for x in [tl, tr]) and not any(
#             x == wrong for x in [br, bl]
#         ):
#             expert_action = d
#
#         elif hitting != cls.NONE_ACTION and tr == br == bl == tl:
#             # Only possible if in 0 vis setting
#             if tr == cls.WRONG_CORNER or last_action == hitting:
#                 if last_action == r:
#                     expert_action = u
#                 elif last_action == u:
#                     expert_action = l
#                 elif last_action == l:
#                     expert_action = d
#                 elif last_action == d:
#                     expert_action = r
#                 else:
#                     raise NotImplementedError()
#             else:
#                 expert_action = last_action
#
#         elif last_action == r and tr == wall:
#             expert_action = u
#
#         elif last_action == u and tl == wall:
#             expert_action = l
#
#         elif last_action == l and bl == wall:
#             expert_action = d
#
#         elif last_action == d and br == wall:
#             expert_action = r
#
#         elif last_action == cls.NONE_ACTION:
#             expert_action = r
#
#         else:
#             expert_action = last_action
#
#         return torch.FloatTensor(
#             [
#                 [
#                     expert_action == u,
#                     expert_action == r,
#                     expert_action == d,
#                     expert_action == l,
#                 ]
#             ]
#         )
#
#     def optimal_average_reward(self, *args, **kwargs):
#         world_radius = (
#             self.world_radius
#             if "world_radius" not in kwargs
#             else kwargs["world_radius"]
#         )
#         visibility_radius = (
#             self.visibility_radius
#             if "visibility_radius" not in kwargs
#             else kwargs["visibility_radius"]
#         )
#
#         tau = 2 * (world_radius - visibility_radius)
#
#         average_steps_needed = 0.25 * (4 * 2 * visibility_radius + 10 * tau)
#
#         return self.SUCCESS_REWARD + self.STEP_PENALTY * (average_steps_needed - 1)
