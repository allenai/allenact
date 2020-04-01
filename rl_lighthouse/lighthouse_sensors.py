import itertools
import typing
from typing import Any, Dict, Optional, List

import gym
import numpy as np
import pandas as pd
import patsy

from rl_base.sensor import Sensor
from rl_base.task import Task
from rl_lighthouse.lighthouse_environment import LightHouseEnvironment


def get_corner_observation(
    env: LightHouseEnvironment,
    view_radius: int,
    view_corner_offsets: Optional[np.array],
):
    if view_corner_offsets is None:
        view_corner_offsets = view_radius * (2 * (env.world_corners > 0) - 1)

    world_corners_offset = env.world_corners + env.world_radius
    multidim_view_corner_indices = np.clip(
        np.reshape(env.current_position, (1, -1))
        + view_corner_offsets
        + env.world_radius,
        a_min=0,
        a_max=2 * env.world_radius,
    )
    flat_view_corner_indices = np.ravel_multi_index(
        np.transpose(multidim_view_corner_indices), env.world_tensor.shape
    )
    view_values = env.world_tensor.reshape(-1)[flat_view_corner_indices]

    last_action = 2 * env.world_dim if env.last_action is None else env.last_action
    on_border_bools = np.concatenate(
        (
            env.current_position == env.world_radius,
            env.current_position == -env.world_radius,
        ),
        axis=0,
    )

    if last_action == 2 * env.world_dim or on_border_bools[last_action]:
        on_border_value = last_action
    elif on_border_bools.any():
        on_border_value = np.argwhere(on_border_bools).reshape(-1)[0]
    else:
        on_border_value = 2 * env.world_dim

    seen_mask = np.array(env.closest_distance_to_corners <= view_radius, dtype=int)
    seen_corner_values = (
        env.world_tensor.reshape(-1)[
            np.ravel_multi_index(
                np.transpose(world_corners_offset), env.world_tensor.shape
            )
        ]
        * seen_mask
    )

    return np.concatenate(
        (
            seen_corner_values + view_values * (1 - seen_mask),
            [on_border_value, last_action],
        ),
        axis=0,
        out=np.zeros((seen_corner_values.shape[0] + 2,), dtype=np.float32,),
    )


class CornerSensor(Sensor[LightHouseEnvironment, Any]):
    def __init__(self, config: Dict[str, Any], *args: Any, **kwargs: Any):
        super().__init__(config, *args, **kwargs)

        self.view_radius = config["view_radius"]
        self.world_dim = config["world_dim"]
        self.view_corner_offsets: Optional[np.ndarray] = None

        self.observation_space = gym.spaces.Box(
            low=min(LightHouseEnvironment.SPACE_LEVELS),
            high=max(LightHouseEnvironment.SPACE_LEVELS),
            shape=(2 ** config["world_dim"],),
            dtype=int,
        )

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return "corner_fixed_radius"

    def _get_observation_space(self) -> gym.spaces.Box:
        return typing.cast(gym.spaces.Box, self.observation_space)

    def get_observation(
        self,
        env: LightHouseEnvironment,
        task: Optional[Task],
        *args: Any,
        **kwargs: Any
    ) -> Any:
        if self.view_corner_offsets is None:
            self.view_corner_offsets = self.view_radius * (
                2 * (env.world_corners > 0) - 1
            )

        return get_corner_observation(
            env=env,
            view_radius=self.view_radius,
            view_corner_offsets=self.view_corner_offsets,
        )


class FactorialDesignCornerSensor(Sensor[LightHouseEnvironment, Any]):
    def __init__(self, config: Dict[str, Any], *args: Any, **kwargs: Any):
        super().__init__(config, *args, **kwargs)

        self.view_radius = config["view_radius"]
        self.world_dim = config["world_dim"]
        self.degree = config["degree"]

        if self.world_dim > 2:
            raise NotImplementedError(
                "When using the `FactorialDesignCornerSensor`,"
                "`world_dim` must be <= 2 due to memory constraints."
                "In the current implementation, creating the design"
                "matrix in the `world_dim == 3` case would require"
                "instantiating a matrix of size ~ 3Mx3M (9 trillion entries)."
            )

        self.view_corner_offsets: Optional[np.ndarray] = None
        # self.world_corners_offset: Optional[List[typing.Tuple[int, ...]]] = None

        self.corner_sensor = CornerSensor(config=config)

        self.variables_and_levels = self._get_variables_and_levels(
            world_dim=self.world_dim
        )
        self._design_mat_formula = self._create_formula(
            variables_and_levels=self._get_variables_and_levels(
                world_dim=self.world_dim
            ),
            degree=self.degree,
        )
        self.single_row_df = pd.DataFrame(
            data=[[0] * len(self.variables_and_levels)],
            columns=[x[0] for x in self.variables_and_levels],
        )
        self._view_tuple_to_design_array: Dict[typing.Tuple[int, ...], np.ndarray] = {}

        (
            design_matrix,
            tuple_to_ind,
        ) = self._create_full_design_matrix_and_tuple_to_ind_dict(
            variables_and_levels=self.variables_and_levels, degree=self.degree
        )

        self.design_matrix = np.array(design_matrix, dtype=bool)
        self.tuple_to_ind = tuple_to_ind

        self.observation_space = gym.spaces.Box(
            low=min(LightHouseEnvironment.SPACE_LEVELS),
            high=max(LightHouseEnvironment.SPACE_LEVELS),
            # shape=(self.output_dim(world_dim=self.world_dim),),
            shape=(
                len(
                    self.view_tuple_to_design_array(
                        (0,) * len(self.variables_and_levels)
                    )
                ),
            ),
            dtype=int,
        )

    def view_tuple_to_design_array(self, view_tuple: typing.Tuple):
        return np.array(
            self.design_matrix[self.tuple_to_ind[view_tuple], :], dtype=np.float32
        )

    @classmethod
    def output_dim(cls, world_dim: int):
        return ((3 if world_dim == 1 else 4) ** (2 ** world_dim)) * (
            2 * world_dim + 1
        ) ** 2

    @classmethod
    def _create_full_design_matrix_and_tuple_to_ind_dict(
        cls, variables_and_levels: List[typing.Tuple[str, List[int]]], degree: int
    ):
        all_tuples = [
            tuple(x)
            for x in itertools.product(*[levels for _, levels in variables_and_levels])
        ]

        tuple_to_ind = {}
        for i, t in enumerate(all_tuples):
            tuple_to_ind[t] = i

        df = pd.DataFrame(
            data=all_tuples, columns=[var_name for var_name, _ in variables_and_levels]
        )

        return (
            1.0
            * patsy.dmatrix(
                cls._create_formula(
                    variables_and_levels=variables_and_levels, degree=degree
                ),
                data=df,
            ),
            tuple_to_ind,
        )

    @classmethod
    def _get_variables_and_levels(self, world_dim: int):
        return (
            [
                ("s{}".format(i), list(range(3 if world_dim == 1 else 4)))
                for i in range(2 ** world_dim)
            ]
            + [("b{}".format(i), list(range(2 * world_dim + 1))) for i in range(1)]
            + [("a{}".format(i), list(range(2 * world_dim + 1))) for i in range(1)]
        )

    @classmethod
    def _create_formula(
        cls, variables_and_levels: List[typing.Tuple[str, List[int]]], degree: int
    ):
        def make_categorial(var_name, levels):
            return "C({}, levels={})".format(var_name, levels)

        if degree == -1:
            return ":".join(
                make_categorial(var_name, levels)
                for var_name, levels in variables_and_levels
            )
        else:
            return "({})**{}".format(
                "+".join(
                    make_categorial(var_name, levels)
                    for var_name, levels in variables_and_levels
                ),
                degree,
            )

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return "corner_fixed_radius_categorical"

    def _get_observation_space(self) -> gym.spaces.Box:
        return typing.cast(gym.spaces.Box, self.observation_space)

    def get_observation(
        self,
        env: LightHouseEnvironment,
        task: Optional[Task],
        *args: Any,
        **kwargs: Any
    ) -> Any:
        kwargs["as_tuple"] = True
        view_array = self.corner_sensor.get_observation(env, task, *args, **kwargs)
        return self.view_tuple_to_design_array(tuple(view_array))

        # design_mat_ind = self.tuple_to_ind[tuple(view_array)]

        # return 1.0 * self.design_mat[design_mat_ind, :]
