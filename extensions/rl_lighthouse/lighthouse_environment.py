import copy
import curses
import itertools
import random
import time
import typing
from functools import lru_cache
from typing import Optional, Tuple, Any, List, Union

import numpy as np

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
        sorted(
            combination_to_vec(comb)
            for i in range(world_dim + 1)
            for comb in itertools.combinations(list(range(world_dim)), i)
        )
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
