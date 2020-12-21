from functools import reduce
import numpy as np


def compute_pose(agent_position, agent_rotation):
    pose = np.eye(4, dtype=np.float32)
    pose[:3, 3] = np.array([agent_position[d] for d in 'xyz'])
    pose[:3, :3] = euler_angles_to_matrix(np.deg2rad(
        np.array([agent_rotation[d] for d in 'zyx'])
    ))
    return pose

def state_to_pose(agent_state):
    agent_position = {k : agent_state[k] for k in 'xyz'}
    agent_rotation = {k : agent_state['rotation'][k] for k in 'xyz'}
    pose = compute_pose(agent_position, agent_rotation)
    return pose
 
# Modified from pytorch3d.transforms
def _axis_angle_rotation(axis, angle):
    cos, sin = np.cos(angle), np.sin(angle)
    one, zero = np.ones_like(angle), np.zeros_like(angle)
    if axis == "X":
        R_flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
    elif axis == "Y":
        R_flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
    elif axis == "Z":
        R_flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)
    return np.stack(R_flat, -1).reshape(angle.shape + (3, 3))

def euler_angles_to_matrix(euler_angles):
    matrices = map(_axis_angle_rotation, 'ZYX', euler_angles)
    return reduce(np.matmul, matrices)

def _angle_from_tan(axis: str, other_axis: str, data, horizontal: bool):
    i1, i2 = {"X": (2, 1), "Y": (0, 2), "Z": (1, 0)}[axis]
    if horizontal:
        i2, i1 = i1, i2
    even = (axis + other_axis) in ["XY", "YZ", "ZX"]
    if horizontal == even:
        return np.arctan2(data[..., i1], data[..., i2])
    return np.arctan2(-data[..., i2], data[..., i1])

def matrix_to_euler_angles(matrix):
    o = (
        _angle_from_tan('X', 'Y', matrix[..., 2], False),
        np.arcsin(matrix[..., 0, 2]),
        _angle_from_tan('Z', 'Y', matrix[..., 0, :], True),
    )
    return np.stack(o, -1)
##