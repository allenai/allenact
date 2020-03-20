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


class CornerSensor(Sensor[LightHouseEnvironment, Any]):
    def __init__(self, config: Dict[str, Any], *args: Any, **kwargs: Any):
        super().__init__(config, *args, **kwargs)

        self.view_radius = config["view_radius"]
        self.world_dim = config["world_dim"]
        self.view_corner_offsets: Optional[np.ndarray] = None
        # self.world_corners_offset: Optional[List[typing.Tuple[int, ...]]] = None

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

        world_corners_offset = env.world_corners + env.world_radius
        multidim_view_corner_indices = np.clip(
            np.reshape(env.current_position, (1, -1))
            + self.view_corner_offsets
            + env.world_radius,
            a_min=0,
            a_max=2 * env.world_radius,
        )
        flat_view_corner_indices = np.ravel_multi_index(
            np.transpose(multidim_view_corner_indices), env.world_tensor.shape
        )
        view_values = env.world_tensor.reshape(-1)[flat_view_corner_indices]

        seen_mask = np.array(
            env.closest_distance_to_corners <= self.view_radius, dtype=int
        )
        seen_corner_values = (
            env.world_tensor.reshape(-1)[
                np.ravel_multi_index(
                    np.transpose(world_corners_offset), env.world_tensor.shape
                )
            ]
            * seen_mask
        )

        on_neg_border = (-1 * (env.current_position == -env.world_radius)) + (
            env.current_position == env.world_radius
        )

        return np.concatenate(
            (view_values, seen_corner_values, on_neg_border),
            axis=0,
            out=np.zeros(
                (
                    view_values.shape[0]
                    + seen_corner_values.shape[0]
                    + on_neg_border.shape[0],
                ),
                dtype=np.float32,
            ),
        )


class FactorialDesignCornerSensor(Sensor[LightHouseEnvironment, Any]):
    def __init__(self, config: Dict[str, Any], *args: Any, **kwargs: Any):
        super().__init__(config, *args, **kwargs)

        self.view_radius = config["view_radius"]
        self.world_dim = config["world_dim"]
        self.degree = config["degree"]

        self.view_corner_offsets: Optional[np.ndarray] = None
        # self.world_corners_offset: Optional[List[typing.Tuple[int, ...]]] = None

        self.corner_sensor = CornerSensor(config=config)

        # (
        #     design_mat,
        #     tuple_to_ind,
        # ) = self._create_full_design_matrix_and_tuple_to_ind_dict(
        #     variables_and_levels=self._get_variables_and_levels(
        #         world_dim=self.world_dim
        #     ),
        #     degree=self.degree
        # )
        # self.design_mat = design_mat
        # self.tuple_to_ind = tuple_to_ind
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
        if view_tuple not in self._view_tuple_to_design_array:
            self.single_row_df.loc[0, :] = view_tuple
            self._view_tuple_to_design_array[view_tuple] = np.array(
                patsy.dmatrix(self._design_mat_formula, data=self.single_row_df,),
                dtype=np.float32,
            ).reshape(-1)
        return self._view_tuple_to_design_array[view_tuple]

    @classmethod
    def output_dim(cls, world_dim: int):
        return (
            ((3 if world_dim == 1 else 4) ** (2 ** world_dim))
            * (3 ** (2 ** world_dim))
            * (3 ** world_dim)
        )

    @classmethod
    def _create_full_design_matrix_and_tuple_to_ind_dict(
        cls, variables_and_levels: List[typing.Tuple[str, List[int]]], degree
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
            + [("m{}".format(i), list(range(3))) for i in range(2 ** world_dim)]
            + [("h{}".format(i), [-1, 0, 1]) for i in range(world_dim)]
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
