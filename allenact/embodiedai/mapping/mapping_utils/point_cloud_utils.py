# MIT License
#
# Original Copyright (c) 2020 Devendra Chaplot
#
# Modified work Copyright (c) 2021 Allen Institute for Artificial Intelligence
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import math
from typing import Optional, Sequence, cast

import numpy as np
import torch


def camera_space_xyz_to_world_xyz(
    camera_space_xyzs: torch.Tensor,
    camera_world_xyz: torch.Tensor,
    rotation: float,
    horizon: float,
) -> torch.Tensor:
    """Transforms xyz coordinates in the camera's coordinate frame to world-
    space (global) xyz frame.

    This code has been adapted from https://github.com/devendrachaplot/Neural-SLAM.

    **IMPORTANT:** We use the conventions from the Unity game engine. In particular:

    * A rotation of 0 corresponds to facing north.
    * Positive rotations correspond to CLOCKWISE rotations. That is a rotation of 90 degrees corresponds
        to facing east. **THIS IS THE OPPOSITE CONVENTION OF THE ONE GENERALLY USED IN MATHEMATICS.**
    * When facing NORTH (rotation==0) moving ahead by 1 meter results in the the z coordinate
        increasing by 1. Moving to the right by 1 meter corresponds to increasing the x coordinate by 1.
         Finally moving upwards by 1 meter corresponds to increasing the y coordinate by 1.
         **Having x,z as the ground plane in this way is common in computer graphics but is different than
         the usual mathematical convention of having z be "up".**
    * The horizon corresponds to how far below the horizontal the camera is facing. I.e. a horizon
        of 30 corresponds to the camera being angled downwards at an angle of 30 degrees.

    # Parameters
    camera_space_xyzs : A 3xN matrix of xyz coordinates in the camera's reference frame.
        Here `x, y, z = camera_space_xyzs[:, i]` should equal the xyz coordinates for the ith point.
    camera_world_xyz : The camera's xyz position in the world reference frame.
    rotation : The world-space rotation (in degrees) of the camera.
    horizon : The horizon (in degrees) of the camera.

    # Returns
    3xN tensor with entry [:, i] is the xyz world-space coordinate corresponding to the camera-space
    coordinate camera_space_xyzs[:, i]
    """
    # Adapted from https://github.com/devendrachaplot/Neural-SLAM.

    # First compute the transformation that points undergo
    # due to the camera's horizon
    psi = -horizon * np.pi / 180
    cos_psi = np.cos(psi)
    sin_psi = np.sin(psi)
    # fmt: off
    horizon_transform = camera_space_xyzs.new(
        [
            [1, 0, 0], # unchanged
            [0, cos_psi, sin_psi],
            [0, -sin_psi, cos_psi,],
        ],
    )
    # fmt: on

    # Next compute the transformation that points undergo
    # due to the agent's rotation about the y-axis
    phi = -rotation * np.pi / 180
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    # fmt: off
    rotation_transform = camera_space_xyzs.new(
        [
            [cos_phi, 0, -sin_phi],
            [0, 1, 0], # unchanged
            [sin_phi, 0, cos_phi],],
    )
    # fmt: on

    # Apply the above transformations
    view_points = (rotation_transform @ horizon_transform) @ camera_space_xyzs

    # Translate the points w.r.t. the camera's position in world space.
    world_points = view_points + camera_world_xyz[:, None]
    return world_points


def depth_frame_to_camera_space_xyz(
    depth_frame: torch.Tensor, mask: Optional[torch.Tensor], fov: float = 90
) -> torch.Tensor:
    """Transforms a input depth map into a collection of xyz points (i.e. a
    point cloud) in the camera's coordinate frame.

    # Parameters
    depth_frame : A square depth map, i.e. an MxM matrix with entry `depth_frame[i, j]` equaling
        the distance from the camera to nearest surface at pixel (i,j).
    mask : An optional boolean mask of the same size (MxM) as the input depth. Only values
        where this mask are true will be included in the returned matrix of xyz coordinates. If
        `None` then no pixels will be masked out (so the returned matrix of xyz points will have
        dimension 3x(M*M)
    fov: The field of view of the camera.

    # Returns

    A 3xN matrix with entry [:, i] equalling a the xyz coordinates (in the camera's coordinate
    frame) of a point in the point cloud corresponding to the input depth frame.
    """
    assert (
        len(depth_frame.shape) == 2 and depth_frame.shape[0] == depth_frame.shape[1]
    ), f"depth has shape {depth_frame.shape}, we only support (N, N) shapes for now."

    resolution = depth_frame.shape[0]
    if mask is None:
        mask = torch.ones_like(depth_frame, dtype=torch.bool)

    # pixel centers
    camera_space_yx_offsets = (
        torch.stack(torch.where(mask))
        + 0.5  # Offset by 0.5 so that we are in the middle of the pixel
    )

    # Subtract center
    camera_space_yx_offsets -= resolution / 2.0

    # Make "up" in y be positive
    camera_space_yx_offsets[0, :] *= -1

    # Put points on the clipping plane
    camera_space_yx_offsets *= (2.0 / resolution) * math.tan((fov / 2) / 180 * math.pi)

    # noinspection PyArgumentList
    camera_space_xyz = torch.cat(
        [
            camera_space_yx_offsets[1:, :],  # This is x
            camera_space_yx_offsets[:1, :],  # This is y
            torch.ones_like(camera_space_yx_offsets[:1, :]),
        ],
        axis=0,
    )

    return camera_space_xyz * depth_frame[mask][None, :]


def depth_frame_to_world_space_xyz(
    depth_frame: torch.Tensor,
    camera_world_xyz: torch.Tensor,
    rotation: float,
    horizon: float,
    fov: float,
):
    """Transforms a input depth map into a collection of xyz points (i.e. a
    point cloud) in the world-space coordinate frame.

    **IMPORTANT:** We use the conventions from the Unity game engine. In particular:

    * A rotation of 0 corresponds to facing north.
    * Positive rotations correspond to CLOCKWISE rotations. That is a rotation of 90 degrees corresponds
        to facing east. **THIS IS THE OPPOSITE CONVENTION OF THE ONE GENERALLY USED IN MATHEMATICS.**
    * When facing NORTH (rotation==0) moving ahead by 1 meter results in the the z coordinate
        increasing by 1. Moving to the right by 1 meter corresponds to increasing the x coordinate by 1.
         Finally moving upwards by 1 meter corresponds to increasing the y coordinate by 1.
         **Having x,z as the ground plane in this way is common in computer graphics but is different than
         the usual mathematical convention of having z be "up".**
    * The horizon corresponds to how far below the horizontal the camera is facing. I.e. a horizon
        of 30 corresponds to the camera being angled downwards at an angle of 30 degrees.

    # Parameters
    depth_frame : A square depth map, i.e. an MxM matrix with entry `depth_frame[i, j]` equaling
        the distance from the camera to nearest surface at pixel (i,j).
    mask : An optional boolean mask of the same size (MxM) as the input depth. Only values
        where this mask are true will be included in the returned matrix of xyz coordinates. If
        `None` then no pixels will be masked out (so the returned matrix of xyz points will have
        dimension 3x(M*M)
    camera_space_xyzs : A 3xN matrix of xyz coordinates in the camera's reference frame.
        Here `x, y, z = camera_space_xyzs[:, i]` should equal the xyz coordinates for the ith point.
    camera_world_xyz : The camera's xyz position in the world reference frame.
    rotation : The world-space rotation (in degrees) of the camera.
    horizon : The horizon (in degrees) of the camera.
    fov: The field of view of the camera.

    # Returns

    A 3xN matrix with entry [:, i] equalling a the xyz coordinates (in the world coordinate
    frame) of a point in the point cloud corresponding to the input depth frame.
    """

    camera_space_xyz = depth_frame_to_camera_space_xyz(
        depth_frame=depth_frame, mask=None, fov=fov
    )

    world_points = camera_space_xyz_to_world_xyz(
        camera_space_xyzs=camera_space_xyz,
        camera_world_xyz=camera_world_xyz,
        rotation=rotation,
        horizon=horizon,
    )

    return world_points.view(3, *depth_frame.shape).permute(1, 2, 0)


def project_point_cloud_to_map(
    xyz_points: torch.Tensor,
    bin_axis: str,
    bins: Sequence[float],
    map_size: int,
    resolution_in_cm: int,
    flip_row_col: bool,
):
    """Bins an input point cloud into a map tensor with the bins equaling the
    channels.

    This code has been adapted from https://github.com/devendrachaplot/Neural-SLAM.

    # Parameters
    xyz_points : (x,y,z) pointcloud(s) as a torch.Tensor of shape (... x height x width x 3).
        All operations are vectorized across the `...` dimensions.
    bin_axis : Either "x", "y", or "z", the axis which should be binned by the values in `bins`.
        If you have generated your point clouds with any of the other functions in the `point_cloud_utils`
        module you almost certainly want this to be "y" as this is the default upwards dimension.
    bins: The values by which to bin along `bin_axis`, see the `bins` parameter of `np.digitize`
        for more info.
    map_size : The axes not specified by `bin_axis` will be be divided by `resolution_in_cm / 100`
        and then rounded to the nearest integer. They are then expected to have their values
        within the interval [0, ..., map_size - 1].
    resolution_in_cm: The resolution_in_cm, in cm, of the map output from this function. Every
        grid square of the map corresponds to a (`resolution_in_cm`x`resolution_in_cm`) square
        in space.
    flip_row_col: Should the rows/cols of the map be flipped? See the 'Returns' section below for more
        info.

    # Returns
    A collection of maps of shape (... x map_size x map_size x (len(bins)+1)), note that bin_axis
    has been moved to the last index of this returned map, the other two axes stay in their original
    order unless `flip_row_col` has been called in which case they are reversed (useful as often
    rows should correspond to y or z instead of x).
    """
    bin_dim = ["x", "y", "z"].index(bin_axis)

    start_shape = xyz_points.shape
    xyz_points = xyz_points.reshape([-1, *start_shape[-3:]])
    num_clouds, h, w, _ = xyz_points.shape

    if not flip_row_col:
        new_order = [i for i in [0, 1, 2] if i != bin_dim] + [bin_dim]
    else:
        new_order = [i for i in [2, 1, 0] if i != bin_dim] + [bin_dim]

    uvw_points = cast(
        torch.Tensor, torch.stack([xyz_points[..., i] for i in new_order], dim=-1)
    )

    num_bins = len(bins) + 1

    isnotnan = ~torch.isnan(xyz_points[..., 0])

    uvw_points_binned: torch.Tensor = torch.cat(
        (
            torch.round(100 * uvw_points[..., :-1] / resolution_in_cm).long(),
            torch.bucketize(
                uvw_points[..., -1:].contiguous(), boundaries=uvw_points.new(bins)
            ),
        ),
        dim=-1,
    )

    maxes = (
        xyz_points.new()
        .long()
        .new([map_size, map_size, num_bins])
        .reshape((1, 1, 1, 3))
    )

    isvalid = torch.logical_and(
        torch.logical_and(
            (uvw_points_binned >= 0).all(-1), (uvw_points_binned < maxes).all(-1),
        ),
        isnotnan,
    )

    uvw_points_binned_with_index_mat = torch.cat(
        (
            torch.repeat_interleave(
                torch.arange(0, num_clouds).to(xyz_points.device), h * w
            ).reshape(-1, 1),
            uvw_points_binned.reshape(-1, 3),
        ),
        dim=1,
    )

    uvw_points_binned_with_index_mat[~isvalid.reshape(-1), :] = 0
    ind = (
        uvw_points_binned_with_index_mat[:, 0] * (map_size * map_size * num_bins)
        + uvw_points_binned_with_index_mat[:, 1] * (map_size * num_bins)
        + uvw_points_binned_with_index_mat[:, 2] * num_bins
        + uvw_points_binned_with_index_mat[:, 3]
    )
    ind[~isvalid.reshape(-1)] = 0
    count = torch.bincount(
        ind.view(-1),
        isvalid.view(-1).long(),
        minlength=num_clouds * map_size * map_size * num_bins,
    )

    return count.view(*start_shape[:-3], map_size, map_size, num_bins)


################
# FOR DEBUGGNG #
################
# The below functions are versions of the above which, because of their reliance on
# numpy functions, cannot use GPU acceleration. These are possibly useful for debugging,
# performance comparisons, or for validating that the above GPU variants work properly.


def _cpu_only_camera_space_xyz_to_world_xyz(
    camera_space_xyzs: np.ndarray,
    camera_world_xyz: np.ndarray,
    rotation: float,
    horizon: float,
):
    # Adapted from https://github.com/devendrachaplot/Neural-SLAM.

    # view_position = 3, world_points = 3 x N
    # NOTE: camera_position is not equal to agent_position!!

    # First compute the transformation that points undergo
    # due to the camera's horizon
    psi = -horizon * np.pi / 180
    cos_psi = np.cos(psi)
    sin_psi = np.sin(psi)
    # fmt: off
    horizon_transform = np.array(
        [
            [1, 0, 0], # unchanged
            [0, cos_psi, sin_psi],
            [0, -sin_psi, cos_psi,],
        ],
        np.float64,
    )
    # fmt: on

    # Next compute the transformation that points undergo
    # due to the agent's rotation about the y-axis
    phi = -rotation * np.pi / 180
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    # fmt: off
    rotation_transform = np.array(
        [
            [cos_phi, 0, -sin_phi],
            [0, 1, 0], # unchanged
            [sin_phi, 0, cos_phi],],
        np.float64,
    )
    # fmt: on

    # Apply the above transformations
    view_points = (rotation_transform @ horizon_transform) @ camera_space_xyzs

    # Translate the points w.r.t. the camera's position in world space.
    world_points = view_points + camera_world_xyz[:, None]
    return world_points


def _cpu_only_depth_frame_to_camera_space_xyz(
    depth_frame: np.ndarray, mask: Optional[np.ndarray], fov: float = 90
):
    """"""
    assert (
        len(depth_frame.shape) == 2 and depth_frame.shape[0] == depth_frame.shape[1]
    ), f"depth has shape {depth_frame.shape}, we only support (N, N) shapes for now."

    resolution = depth_frame.shape[0]
    if mask is None:
        mask = np.ones(depth_frame.shape, dtype=bool)

    # pixel centers
    camera_space_yx_offsets = (
        np.stack(np.where(mask))
        + 0.5  # Offset by 0.5 so that we are in the middle of the pixel
    )

    # Subtract center
    camera_space_yx_offsets -= resolution / 2.0

    # Make "up" in y be positive
    camera_space_yx_offsets[0, :] *= -1

    # Put points on the clipping plane
    camera_space_yx_offsets *= (2.0 / resolution) * math.tan((fov / 2) / 180 * math.pi)

    camera_space_xyz = np.concatenate(
        [
            camera_space_yx_offsets[1:, :],  # This is x
            camera_space_yx_offsets[:1, :],  # This is y
            np.ones_like(camera_space_yx_offsets[:1, :]),
        ],
        axis=0,
    )

    return camera_space_xyz * depth_frame[mask][None, :]


def _cpu_only_depth_frame_to_world_space_xyz(
    depth_frame: np.ndarray,
    camera_world_xyz: np.ndarray,
    rotation: float,
    horizon: float,
    fov: float,
):
    camera_space_xyz = _cpu_only_depth_frame_to_camera_space_xyz(
        depth_frame=depth_frame, mask=None, fov=fov
    )

    world_points = _cpu_only_camera_space_xyz_to_world_xyz(
        camera_space_xyzs=camera_space_xyz,
        camera_world_xyz=camera_world_xyz,
        rotation=rotation,
        horizon=horizon,
    )

    return world_points.reshape((3, *depth_frame.shape)).transpose((1, 2, 0))


def _cpu_only_project_point_cloud_to_map(
    xyz_points: np.ndarray,
    bin_axis: str,
    bins: Sequence[float],
    map_size: int,
    resolution_in_cm: int,
    flip_row_col: bool,
):
    """Bins points into  bins.

    Adapted from https://github.com/devendrachaplot/Neural-SLAM.

    # Parameters
    xyz_points : (x,y,z) point clouds as a np.ndarray of shape (... x height x width x 3). (x,y,z)
        should be coordinates specified in meters.
    bin_axis : Either "x", "y", or "z", the axis which should be binned by the values in `bins`
    bins: The values by which to bin along `bin_axis`, see the `bins` parameter of `np.digitize`
        for more info.
    map_size : The axes not specified by `bin_axis` will be be divided by `resolution_in_cm / 100`
        and then rounded to the nearest integer. They are then expected to have their values
        within the interval [0, ..., map_size - 1].
    resolution_in_cm: The resolution_in_cm, in cm, of the map output from this function. Every
        grid square of the map corresponds to a (`resolution_in_cm`x`resolution_in_cm`) square
        in space.
    flip_row_col: Should the rows/cols of the map be flipped

    # Returns
    A collection of maps of shape (... x map_size x map_size x (len(bins)+1)), note that bin_axis
    has been moved to the last index of this returned map, the other two axes stay in their original
    order unless `flip_row_col` has been called in which case they are reversed (useful if you give
    points as often rows should correspond to y or z instead of x).
    """
    bin_dim = ["x", "y", "z"].index(bin_axis)

    start_shape = xyz_points.shape
    xyz_points = xyz_points.reshape([-1, *start_shape[-3:]])
    num_clouds, h, w, _ = xyz_points.shape

    if not flip_row_col:
        new_order = [i for i in [0, 1, 2] if i != bin_dim] + [bin_dim]
    else:
        new_order = [i for i in [2, 1, 0] if i != bin_dim] + [bin_dim]

    uvw_points: np.ndarray = np.stack([xyz_points[..., i] for i in new_order], axis=-1)

    num_bins = len(bins) + 1

    isnotnan = ~np.isnan(xyz_points[..., 0])

    uvw_points_binned = np.concatenate(
        (
            np.round(100 * uvw_points[..., :-1] / resolution_in_cm).astype(np.int32),
            np.digitize(uvw_points[..., -1:], bins=bins).astype(np.int32),
        ),
        axis=-1,
    )

    maxes = np.array([map_size, map_size, num_bins]).reshape((1, 1, 1, 3))

    isvalid = np.logical_and.reduce(
        (
            (uvw_points_binned >= 0).all(-1),
            (uvw_points_binned < maxes).all(-1),
            isnotnan,
        )
    )

    uvw_points_binned_with_index_mat = np.concatenate(
        (
            np.repeat(np.arange(0, num_clouds), h * w).reshape(-1, 1),
            uvw_points_binned.reshape(-1, 3),
        ),
        axis=1,
    )

    uvw_points_binned_with_index_mat[~isvalid.reshape(-1), :] = 0
    ind = np.ravel_multi_index(
        uvw_points_binned_with_index_mat.transpose(),
        (num_clouds, map_size, map_size, num_bins),
    )
    ind[~isvalid.reshape(-1)] = 0
    count = np.bincount(
        ind.ravel(),
        isvalid.ravel().astype(np.int32),
        minlength=num_clouds * map_size * map_size * num_bins,
    )

    return count.reshape([*start_shape[:-3], map_size, map_size, num_bins])
