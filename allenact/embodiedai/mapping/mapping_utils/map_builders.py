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

import random
from typing import Optional, Sequence, Union, Dict

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from allenact.embodiedai.mapping.mapping_utils.point_cloud_utils import (
    depth_frame_to_world_space_xyz,
    project_point_cloud_to_map,
)


class BinnedPointCloudMapBuilder(object):
    """Class used to iteratively construct a map of "free space" based on input
    depth maps (i.e. pointclouds).

    Adapted from https://github.com/devendrachaplot/Neural-SLAM

    This class can be used to (iteratively) construct a metric map of free space in an environment as
    an agent moves around. After every step the agent takes, you should call the `update` function and
    pass the agent's egocentric depth image along with the agent's new position. This depth map will
    be converted into a pointcloud, binned along the up/down axis, and then projected
    onto a 3-dimensional tensor of shape (HxWxC) whose where HxW represent the ground plane
    and where C equals the number of bins the up-down coordinate was binned into. This 3d map counts the
    number of points in each bin. Thus a lack of points within a region can be used to infer that
    that region is free space.

    # Attributes

    fov : FOV of the camera used to produce the depth images given when calling `update`.
    vision_range_in_map_units : The maximum distance (in number of rows/columns) that will
        be updated when calling `update`, points outside of this map vision range are ignored.
    map_size_in_cm : Total map size in cm.
    resolution_in_cm : Number of cm per row/column in the map.
    height_bins : The bins used to bin the up-down coordinate (for us the y-coordinate). For example,
        if `height_bins = [0.1, 1]` then
        all y-values < 0.1 will be mapped to 0, all y values in [0.1, 1) will be mapped to 1, and
        all y-values >= 1 will be mapped to 2.
        **Importantly:** these y-values will first be recentered by the `min_xyz` value passed when
        calling `reset(...)`.
    device : A `torch.device` on which to run computations. If this device is a GPU you can potentially
        obtain significant speed-ups.
    """

    def __init__(
        self,
        fov: float,
        vision_range_in_cm: int,
        map_size_in_cm: int,
        resolution_in_cm: int,
        height_bins: Sequence[float],
        return_egocentric_local_context: bool = False,
        device: torch.device = torch.device("cpu"),
    ):
        assert vision_range_in_cm % resolution_in_cm == 0

        self.fov = fov
        self.vision_range_in_map_units = vision_range_in_cm // resolution_in_cm
        self.map_size_in_cm = map_size_in_cm
        self.resolution_in_cm = resolution_in_cm
        self.height_bins = height_bins
        self.device = device
        self.return_egocentric_local_context = return_egocentric_local_context

        self.binned_point_cloud_map = np.zeros(
            (
                self.map_size_in_cm // self.resolution_in_cm,
                self.map_size_in_cm // self.resolution_in_cm,
                len(self.height_bins) + 1,
            ),
            dtype=np.float32,
        )

        self.min_xyz: Optional[np.ndarray] = None

    def update(
        self,
        depth_frame: np.ndarray,
        camera_xyz: np.ndarray,
        camera_rotation: float,
        camera_horizon: float,
    ) -> Dict[str, np.ndarray]:
        """Updates the map with the input depth frame from the agent.

        See the `allenact.embodiedai.mapping.mapping_utils.point_cloud_utils.project_point_cloud_to_map`
        function for more information input parameter definitions. **We assume that the input
        `depth_frame` has depths recorded in meters**.

        # Returns
        Let `map_size = self.map_size_in_cm // self.resolution_in_cm`. Returns a dictionary with keys-values:

        * `"egocentric_update"` - A tensor of shape
            `(vision_range_in_map_units)x(vision_range_in_map_units)x(len(self.height_bins) + 1)` corresponding
            to the binned pointcloud after having been centered on the agent and rotated so that
            points ahead of the agent correspond to larger row indices and points further to the right of the agent
            correspond to larger column indices. Note that by "centered" we mean that one can picture
             the agent as being positioned at (0, vision_range_in_map_units/2) and facing downward. Each entry in this tensor
             is a count equaling the number of points in the pointcloud that, once binned, fell into this
            entry. This is likely the output you want to use if you want to build a model to predict free space from an image.
        * `"allocentric_update"` - A `(map_size)x(map_size)x(len(self.height_bins) + 1)` corresponding
            to `"egocentric_update"` but rotated to the world-space coordinates. This `allocentric_update`
             is what is used to update the internally stored representation of the map.
        *  `"map"` -  A `(map_size)x(map_size)x(len(self.height_bins) + 1)` tensor corresponding
            to the sum of all `"allocentric_update"` values since the last `reset()`.
        ```
        """
        with torch.no_grad():
            assert self.min_xyz is not None, "Please call `reset` before `update`."

            camera_xyz = (
                torch.from_numpy(camera_xyz - self.min_xyz).float().to(self.device)
            )

            try:
                depth_frame = torch.from_numpy(depth_frame).to(self.device)
            except ValueError:
                depth_frame = torch.from_numpy(depth_frame.copy()).to(self.device)

            depth_frame[
                depth_frame
                > self.vision_range_in_map_units * self.resolution_in_cm / 100
            ] = np.NaN

            world_space_point_cloud = depth_frame_to_world_space_xyz(
                depth_frame=depth_frame,
                camera_world_xyz=camera_xyz,
                rotation=camera_rotation,
                horizon=camera_horizon,
                fov=self.fov,
            )

            world_binned_map_update = project_point_cloud_to_map(
                xyz_points=world_space_point_cloud,
                bin_axis="y",
                bins=self.height_bins,
                map_size=self.binned_point_cloud_map.shape[0],
                resolution_in_cm=self.resolution_in_cm,
                flip_row_col=True,
            )

            # Center the cloud on the agent
            recentered_point_cloud = world_space_point_cloud - (
                torch.FloatTensor([1.0, 0.0, 1.0]).to(self.device) * camera_xyz
            ).reshape((1, 1, 3))
            # Rotate the cloud so that positive-z is the direction the agent is looking
            theta = (
                np.pi * camera_rotation / 180
            )  # No negative since THOR rotations are already backwards
            cos_theta = np.cos(theta)
            sin_theta = np.sin(theta)
            rotation_transform = torch.FloatTensor(
                [
                    [cos_theta, 0, -sin_theta],
                    [0, 1, 0],  # unchanged
                    [sin_theta, 0, cos_theta],
                ]
            ).to(self.device)
            rotated_point_cloud = recentered_point_cloud @ rotation_transform.T
            xoffset = (self.map_size_in_cm / 100) / 2
            agent_centric_point_cloud = rotated_point_cloud + torch.FloatTensor(
                [xoffset, 0, 0]
            ).to(self.device)

            allocentric_update_numpy = world_binned_map_update.cpu().numpy()
            self.binned_point_cloud_map = (
                self.binned_point_cloud_map + allocentric_update_numpy
            )

            agent_centric_binned_map = project_point_cloud_to_map(
                xyz_points=agent_centric_point_cloud,
                bin_axis="y",
                bins=self.height_bins,
                map_size=self.binned_point_cloud_map.shape[0],
                resolution_in_cm=self.resolution_in_cm,
                flip_row_col=True,
            )
            vr = self.vision_range_in_map_units
            vr_div_2 = self.vision_range_in_map_units // 2
            width_div_2 = agent_centric_binned_map.shape[1] // 2
            agent_centric_binned_map = agent_centric_binned_map[
                :vr, (width_div_2 - vr_div_2) : (width_div_2 + vr_div_2), :
            ]

            to_return = {
                "egocentric_update": agent_centric_binned_map.cpu().numpy(),
                "allocentric_update": allocentric_update_numpy,
                "map": self.binned_point_cloud_map,
            }

            if self.return_egocentric_local_context:
                # See the update function of the semantic map sensor for in depth comments regarding the below
                # Essentially we are simply rotating the full map into the orientation of the agent and then
                # selecting a smaller region around the agent.
                theta = -np.pi * camera_rotation / 180
                cos_theta = np.cos(theta)
                sin_theta = np.sin(theta)
                rot_mat = torch.FloatTensor(
                    [[cos_theta, -sin_theta], [sin_theta, cos_theta]]
                ).to(self.device)

                move_back_offset = (
                    -0.5
                    * (self.vision_range_in_map_units * self.resolution_in_cm / 100)
                ) * (
                    rot_mat
                    @ torch.tensor(
                        [0, 1], dtype=torch.float, device=self.device
                    ).unsqueeze(-1)
                )

                map_size = self.binned_point_cloud_map.shape[0]
                scaler = 2 * (100 / (self.resolution_in_cm * map_size))
                offset_to_center_the_agent = (
                    scaler
                    * (
                        torch.tensor(
                            [camera_xyz[0], camera_xyz[2],],
                            dtype=torch.float,
                            device=self.device,
                        ).unsqueeze(-1)
                        + move_back_offset
                    )
                    - 1
                )
                offset_to_top_of_image = rot_mat @ torch.FloatTensor(
                    [0, 1.0]
                ).unsqueeze(1).to(self.device)
                rotation_and_translate_mat = torch.cat(
                    (rot_mat, offset_to_top_of_image + offset_to_center_the_agent,),
                    dim=1,
                )

                full_map_tensor = (
                    torch.tensor(
                        self.binned_point_cloud_map,
                        dtype=torch.float,
                        device=self.device,
                    )
                    .unsqueeze(0)
                    .permute(0, 3, 1, 2)
                )
                full_ego_map = (
                    F.grid_sample(
                        full_map_tensor,
                        F.affine_grid(
                            rotation_and_translate_mat.to(self.device).unsqueeze(0),
                            full_map_tensor.shape,
                            align_corners=False,
                        ),
                        align_corners=False,
                    )
                    .squeeze(0)
                    .permute(1, 2, 0)
                )

                egocentric_local_context = full_ego_map[
                    :vr, (width_div_2 - vr_div_2) : (width_div_2 + vr_div_2), :
                ]

                to_return[
                    "egocentric_local_context"
                ] = egocentric_local_context.cpu().numpy()

            return to_return

    def reset(self, min_xyz: np.ndarray):
        """Reset the map.

        Resets the internally stored map.

        # Parameters
        min_xyz : An array of size (3,) corresponding to the minimum possible x, y, and z values that will be observed
            as a point in a pointcloud when calling `.update(...)`. The (world-space) maps returned by calls to `update`
            will have been normalized so the (0,0,:) entry corresponds to these minimum values.
        """
        self.min_xyz = min_xyz
        self.binned_point_cloud_map = np.zeros_like(self.binned_point_cloud_map)


class ObjectHull2d:
    def __init__(
        self,
        object_id: str,
        object_type: str,
        hull_points: Union[np.ndarray, Sequence[Sequence[float]]],
    ):
        """A class used to represent 2d convex hulls of objects when projected
        to the ground plane.

        # Parameters
        object_id : A unique id for the object.
        object_type : The type of the object.
        hull_points : A Nx2 matrix with `hull_points[:, 0]` being the x coordinates and `hull_points[:, 1]` being
            the `z` coordinates (this is using the Unity game engine conventions where the `y` axis is up/down).
        """
        self.object_id = object_id
        self.object_type = object_type
        self.hull_points = (
            hull_points
            if isinstance(hull_points, np.ndarray)
            else np.array(hull_points)
        )


class SemanticMapBuilder(object):
    """Class used to iteratively construct a semantic map based on input depth
    maps (i.e. pointclouds).

    Adapted from https://github.com/devendrachaplot/Neural-SLAM

    This class can be used to (iteratively) construct a semantic map of objects in the environment.

    This map is similar to that generated by `BinnedPointCloudMapBuilder` (see its documentation for
    more information) but the various channels correspond to different object types. Thus
    if the `(i,j,k)` entry of a map generated by this function is `True`, this means that an
    object of type `k` is present in position `i,j` in the map. In particular, by "present" we mean that,
    after projecting the object to the ground plane and taking the convex hull of the resulting
    2d object, a non-trivial portion of this convex hull overlaps the `i,j` position.

    For attribute information, see the documentation of the `BinnedPointCloudMapBuilder` class. The
    only attribute present in this class that is not present in `BinnedPointCloudMapBuilder` is
    `ordered_object_types` which corresponds to a list of unique object types where
    object type `ordered_object_types[i]` will correspond to the `i`th channel of the map
    generated by this class.
    """

    def __init__(
        self,
        fov: float,
        vision_range_in_cm: int,
        map_size_in_cm: int,
        resolution_in_cm: int,
        ordered_object_types: Sequence[str],
        device: torch.device = torch.device("cpu"),
    ):
        self.fov = fov
        self.vision_range_in_map_units = vision_range_in_cm // resolution_in_cm
        self.map_size_in_cm = map_size_in_cm
        self.resolution_in_cm = resolution_in_cm
        self.ordered_object_types = tuple(ordered_object_types)
        self.device = device

        self.object_type_to_index = {
            ot: i for i, ot in enumerate(self.ordered_object_types)
        }

        self.ground_truth_semantic_map = np.zeros(
            (
                self.map_size_in_cm // self.resolution_in_cm,
                self.map_size_in_cm // self.resolution_in_cm,
                len(self.ordered_object_types),
            ),
            dtype=np.uint8,
        )
        self.explored_mask = np.zeros(
            (
                self.map_size_in_cm // self.resolution_in_cm,
                self.map_size_in_cm // self.resolution_in_cm,
                1,
            ),
            dtype=bool,
        )

        self.min_xyz: Optional[np.ndarray] = None

    @staticmethod
    def randomly_color_semantic_map(
        map: Union[np.ndarray, torch.Tensor], threshold: float = 0.5, seed: int = 1
    ) -> np.ndarray:
        if not isinstance(map, np.ndarray):
            map = np.array(map)

        rnd = random.Random(seed)
        semantic_int_mat = (
            (map >= threshold)
            * np.array(list(range(1, map.shape[-1] + 1))).reshape((1, 1, -1))
        ).max(-1)
        # noinspection PyTypeChecker
        return np.uint8(
            np.array(
                [(0, 0, 0)]
                + [
                    tuple(rnd.randint(0, 256) for _ in range(3))
                    for _ in range(map.shape[-1])
                ]
            )[semantic_int_mat]
        )

    def _xzs_to_colrows(self, xzs: np.ndarray):
        height, width, _ = self.ground_truth_semantic_map.shape
        return np.clip(
            np.int32(
                (
                    (100 / self.resolution_in_cm)
                    * (xzs - np.array([[self.min_xyz[0], self.min_xyz[2]]]))
                )
            ),
            a_min=0,
            a_max=np.array(
                [width - 1, height - 1]
            ),  # width then height as we're returns cols then rows
        )

    def build_ground_truth_map(self, object_hulls: Sequence[ObjectHull2d]):
        self.ground_truth_semantic_map.fill(0)

        height, width, _ = self.ground_truth_semantic_map.shape
        for object_hull in object_hulls:
            ot = object_hull.object_type

            if ot in self.object_type_to_index:
                ind = self.object_type_to_index[ot]

                self.ground_truth_semantic_map[
                    :, :, ind : (ind + 1)
                ] = cv2.fillConvexPoly(
                    img=np.array(
                        self.ground_truth_semantic_map[:, :, ind : (ind + 1)],
                        dtype=np.uint8,
                    ),
                    points=self._xzs_to_colrows(np.array(object_hull.hull_points)),
                    color=255,
                )

    def update(
        self,
        depth_frame: np.ndarray,
        camera_xyz: np.ndarray,
        camera_rotation: float,
        camera_horizon: float,
    ) -> Dict[str, np.ndarray]:
        """Updates the map with the input depth frame from the agent.

        See the documentation for `BinnedPointCloudMapBuilder.update`,
        the inputs and outputs are similar except that channels are used
        to represent the presence/absence of objects of given types.
        Unlike `BinnedPointCloudMapBuilder.update`, this function also
        returns two masks with keys `"egocentric_mask"` and `"mask"`
        that can be used to determine what portions of the map have been
        observed by the agent so far in the egocentric and world-space
        reference frames respectively.
        """
        with torch.no_grad():
            assert self.min_xyz is not None

            camera_xyz = torch.from_numpy(camera_xyz - self.min_xyz).to(self.device)
            map_size = self.ground_truth_semantic_map.shape[0]

            depth_frame = torch.from_numpy(depth_frame).to(self.device)
            depth_frame[
                depth_frame
                > self.vision_range_in_map_units * self.resolution_in_cm / 100
            ] = np.NaN

            world_space_point_cloud = depth_frame_to_world_space_xyz(
                depth_frame=depth_frame,
                camera_world_xyz=camera_xyz,
                rotation=camera_rotation,
                horizon=camera_horizon,
                fov=self.fov,
            )

            world_newly_explored = (
                project_point_cloud_to_map(
                    xyz_points=world_space_point_cloud,
                    bin_axis="y",
                    bins=[],
                    map_size=map_size,
                    resolution_in_cm=self.resolution_in_cm,
                    flip_row_col=True,
                )
                > 0.001
            )
            world_update_and_mask = torch.cat(
                (
                    torch.logical_and(
                        torch.from_numpy(self.ground_truth_semantic_map).to(
                            self.device
                        ),
                        world_newly_explored,
                    ),
                    world_newly_explored,
                ),
                dim=-1,
            ).float()
            world_update_and_mask_for_sample = world_update_and_mask.unsqueeze(
                0
            ).permute(0, 3, 1, 2)

            # We now use grid sampling to rotate world_update_for_sample into the egocentric coordinate
            # frame of the agent so that the agent's forward direction is downwards in the tensor
            # (and it's right side is to the right in the image, this means that right/left
            # when taking the perspective of the agent in the image). This convention aligns with
            # what's expected by grid_sample where +x corresponds to +cols and +z corresponds to +rows.
            # Here also the rows/cols have been normalized so that the center of the image is at (0,0)
            # and the bottom right is at (1,1).

            # Mentally you can think of the output from the F.affine_grid function as you wanting
            # rotating/translating an axis-aligned square on the image-to-be-sampled and then
            # copying whatever is in this square to a new image. Note that the translation always
            # happens in the global reference frame after the rotation. We'll start by rotating
            # the square so that the the agent's z direction is downwards in the image.
            # Since the global axis of the map and the grid sampling are aligned, this requires
            # rotating the square by the rotation of the agent. As rotation is negative the usual
            # standard in THOR, we need to negate the rotation of the agent.
            theta = -np.pi * camera_rotation / 180

            # Here form the rotation matrix
            cos_theta = np.cos(theta)
            sin_theta = np.sin(theta)
            rot_mat = torch.FloatTensor(
                [[cos_theta, -sin_theta], [sin_theta, cos_theta]]
            ).to(self.device)

            # Now we need to figure out the translation. For an intuitive understanding, we break this
            # translation into two different "offsets". The first offset centers the square on the
            # agent's current location:
            scaler = 2 * (100 / (self.resolution_in_cm * map_size))
            offset_to_center_the_agent = (
                scaler
                * torch.FloatTensor([camera_xyz[0], camera_xyz[2]])
                .unsqueeze(-1)
                .to(self.device)
                - 1
            )
            # The second offset moves the square in the direction of the agent's z direction
            # so that the output image will have the agent's view starting directly at the
            # top of the image.
            offset_to_top_of_image = rot_mat @ torch.FloatTensor([0, 1.0]).unsqueeze(
                1
            ).to(self.device)
            rotation_and_translate_mat = torch.cat(
                (rot_mat, offset_to_top_of_image + offset_to_center_the_agent,), dim=1,
            )

            ego_update_and_mask = F.grid_sample(
                world_update_and_mask_for_sample.to(self.device),
                F.affine_grid(
                    rotation_and_translate_mat.to(self.device).unsqueeze(0),
                    world_update_and_mask_for_sample.shape,
                    align_corners=False,
                ),
                align_corners=False,
            )

            # All that's left now is to crop out the portion of the transformed tensor that we actually
            # care about (i.e. the portion corresponding to the agent's `self.vision_range_in_map_units`.
            vr = self.vision_range_in_map_units
            half_vr = vr // 2
            center = self.map_size_in_cm // (2 * self.resolution_in_cm)
            cropped = ego_update_and_mask[
                :, :, :vr, (center - half_vr) : (center + half_vr)
            ]

            np.logical_or(
                self.explored_mask,
                world_newly_explored.cpu().numpy(),
                out=self.explored_mask,
            )

            return {
                "egocentric_update": cropped[0, :-1].permute(1, 2, 0).cpu().numpy(),
                "egocentric_mask": (cropped[0, -1:].view(vr, vr, 1) > 0.001)
                .cpu()
                .numpy(),
                "explored_mask": np.array(self.explored_mask),
                "map": np.logical_and(
                    self.explored_mask, (self.ground_truth_semantic_map > 0)
                ),
            }

    def reset(self, min_xyz: np.ndarray, object_hulls: Sequence[ObjectHull2d]):
        """Reset the map.

        Resets the internally stored map.

        # Parameters
        min_xyz : An array of size (3,) corresponding to the minimum possible x, y, and z values that will be observed
            as a point in a pointcloud when calling `.update(...)`. The (world-space) maps returned by calls to `update`
            will have been normalized so the (0,0,:) entry corresponds to these minimum values.
        object_hulls : The object hulls corresponding to objects in the scene. These will be used to
            construct the map.
        """
        self.min_xyz = min_xyz
        self.build_ground_truth_map(object_hulls=object_hulls)
