from functools import reduce
import numpy as np


def compute_pose(agent_position, agent_rotation):
    pose = np.eye(4, dtype=np.float32)
    pose[:3, 3] = np.array([agent_position[d] for d in 'xyz'])
    pose[:3, :3] = euler_angles_to_matrix(np.deg2rad(
        np.array([agent_rotation[d] for d in 'zyx'])
    ), 'ZYX')
    return pose

def state_to_pose(agent_state):
    agent_position = {k : agent_state[k] for k in 'xyz'}
    agent_rotation = {k : agent_state['rotation'][k] for k in 'xyz'}
    pose = compute_pose(agent_position, agent_rotation)
    return pose
 
def pose_to_state(pose):
    agent_position = pose[:3, 3]
    agent_rotation = np.rad2deg(matrix_to_euler_angles(pose[:3, :3], 'ZYX'))
    return agent_position, agent_rotation

# Modified from pytorch3d.transforms
def _axis_angle_rotation(axis: str, angle):
    cos = np.cos(angle)
    sin = np.sin(angle)
    one = np.ones_like(angle)
    zero = np.zeros_like(angle)
    if axis == "X":
        R_flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
    if axis == "Y":
        R_flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
    if axis == "Z":
        R_flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)
    return np.stack(R_flat, -1).reshape(angle.shape + (3, 3))

def euler_angles_to_matrix(euler_angles, convention: str):
    matrices = map(_axis_angle_rotation, convention, euler_angles)
    return reduce(np.matmul, matrices)

def _angle_from_tan(axis: str, other_axis: str, data, horizontal: bool, tait_bryan: bool):
    i1, i2 = {"X": (2, 1), "Y": (0, 2), "Z": (1, 0)}[axis]
    if horizontal:
        i2, i1 = i1, i2
    even = (axis + other_axis) in ["XY", "YZ", "ZX"]
    if horizontal == even:
        return np.arctan2(data[..., i1], data[..., i2])
    if tait_bryan:
        return np.arctan2(-data[..., i2], data[..., i1])
    return np.arctan2(data[..., i2], -data[..., i1])

def _index_from_letter(letter: str):
    if letter == "X":
        return 0
    if letter == "Y":
        return 1
    if letter == "Z":
        return 2

def matrix_to_euler_angles(matrix, convention: str):
    i0 = _index_from_letter(convention[0])
    i2 = _index_from_letter(convention[2])
    tait_bryan = i0 != i2
    if tait_bryan:
        central_angle = np.arcsin(
            matrix[..., i0, i2] * (-1.0 if i0 - i2 in [-1, 2] else 1.0)
        )
    else:
        central_angle = np.arccos(matrix[..., i0, i0])
    o = (
        _angle_from_tan(
            convention[0], convention[1], matrix[..., i2], False, tait_bryan
        ),
        central_angle,
        _angle_from_tan(
            convention[2], convention[1], matrix[..., i0, :], True, tait_bryan
        ),
    )
    return np.stack(o, -1)
##
