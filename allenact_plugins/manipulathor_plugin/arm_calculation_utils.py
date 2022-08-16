"""Utility classes and functions for calculating the arm relative and absolute
position."""
from typing import Dict

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

from allenact.utils.system import get_logger


def state_dict_to_tensor(state: Dict):
    result = []
    if "position" in state:
        result += [
            state["position"]["x"],
            state["position"]["y"],
            state["position"]["z"],
        ]
    if "rotation" in state:
        result += [
            state["rotation"]["x"],
            state["rotation"]["y"],
            state["rotation"]["z"],
        ]
    return torch.Tensor(result)


def diff_position(state_goal, state_curr, absolute: bool = True):
    p1 = state_goal["position"]
    p2 = state_curr["position"]
    if absolute:
        result = {k: abs(p1[k] - p2[k]) for k in p1.keys()}
    else:
        result = {k: (p1[k] - p2[k]) for k in p1.keys()}
    return result


def coord_system_transform(position: Dict, coord_system: str):
    assert coord_system in [
        "xyz_unsigned",
        "xyz_signed",
        "polar_radian",
        "polar_trigo",
    ]

    if "xyz" in coord_system:
        result = [
            position["x"],
            position["y"],
            position["z"],
        ]
        result = torch.Tensor(result)
        if coord_system == "xyz_unsigned":
            return torch.abs(result)
        else:  # xyz_signed
            return result

    else:
        hxy = np.hypot(position["x"], position["y"])
        r = np.hypot(hxy, position["z"])
        el = np.arctan2(position["z"], hxy)  # elevation angle: [-pi/2, pi/2]
        az = np.arctan2(position["y"], position["x"])  # azimuthal angle: [-pi, pi]

        if coord_system == "polar_radian":
            result = [
                r,
                el / (0.5 * np.pi),
                az / np.pi,
            ]  # normalize to [-1, 1]
            return torch.Tensor(result)
        else:  # polar_trigo
            result = [
                r,
                np.cos(el),
                np.sin(el),
                np.cos(az),
                np.sin(az),
            ]
            return torch.Tensor(result)


def position_rotation_to_matrix(position, rotation):
    result = np.zeros((4, 4))
    r = R.from_euler("xyz", [rotation["x"], rotation["y"], rotation["z"]], degrees=True)
    result[:3, :3] = r.as_matrix()
    result[3, 3] = 1
    result[:3, 3] = [position["x"], position["y"], position["z"]]
    return result


def inverse_rot_trans_matrix(mat):
    mat = np.linalg.inv(mat)
    return mat


def matrix_to_position_rotation(matrix):
    result = {"position": None, "rotation": None}
    rotation = R.from_matrix(matrix[:3, :3]).as_euler("xyz", degrees=True)
    rotation_dict = {"x": rotation[0], "y": rotation[1], "z": rotation[2]}
    result["rotation"] = rotation_dict
    position = matrix[:3, 3]
    result["position"] = {"x": position[0], "y": position[1], "z": position[2]}
    return result


def find_closest_inverse(deg, use_cache):
    if use_cache:
        for k in _saved_inverse_rotation_mats.keys():
            if abs(k - deg) < 5:
                return _saved_inverse_rotation_mats[k]
    # if it reaches here it means it had not calculated the degree before
    rotation = R.from_euler("xyz", [0, deg, 0], degrees=True)
    result = rotation.as_matrix()
    inverse = inverse_rot_trans_matrix(result)
    if use_cache:
        get_logger().warning(f"Had to calculate the matrix for {deg}")
    return inverse


def calc_inverse(deg):
    rotation = R.from_euler("xyz", [0, deg, 0], degrees=True)
    result = rotation.as_matrix()
    inverse = inverse_rot_trans_matrix(result)
    return inverse


_saved_inverse_rotation_mats = {i: calc_inverse(i) for i in range(0, 360, 45)}
_saved_inverse_rotation_mats[360] = _saved_inverse_rotation_mats[0]


def world_coords_to_agent_coords(world_obj, agent_state, use_cache=True):
    position = agent_state["position"]
    rotation = agent_state["rotation"]
    agent_translation = [position["x"], position["y"], position["z"]]
    assert abs(rotation["x"]) < 0.01 and abs(rotation["z"]) < 0.01
    inverse_agent_rotation = find_closest_inverse(rotation["y"], use_cache=use_cache)
    obj_matrix = position_rotation_to_matrix(
        world_obj["position"], world_obj["rotation"]
    )
    obj_translation = np.matmul(
        inverse_agent_rotation, (obj_matrix[:3, 3] - agent_translation)
    )
    # add rotation later
    obj_matrix[:3, 3] = obj_translation
    result = matrix_to_position_rotation(obj_matrix)
    return result
