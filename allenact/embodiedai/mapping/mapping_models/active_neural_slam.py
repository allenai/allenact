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
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

from allenact.utils.model_utils import simple_conv_and_linear_weights_init

DEGREES_TO_RADIANS = np.pi / 180.0
RADIANS_TO_DEGREES = 180.0 / np.pi


def _inv_sigmoid(x: torch.Tensor):
    return torch.log(x) - torch.log1p(-x)


class ActiveNeuralSLAM(nn.Module):
    """Active Neural SLAM module.

    This is an implementation of the Active Neural SLAM module
    from:
    ```
    Chaplot, D.S., Gandhi, D., Gupta, S., Gupta, A. and Salakhutdinov, R., 2020.
    Learning To Explore Using Active Neural SLAM.
    In International Conference on Learning Representations (ICLR).
    ```
    Note that this is purely the mapping component and does not include the planning
    components from the above paper.

    This implementation is adapted from `https://github.com/devendrachaplot/Neural-SLAM`,
    we have extended this implementation to allow for an arbitrary number of output map
    channels (enabling semantic mapping).

    At a high level, this model takes as input RGB egocentric images and outputs metric
    map tensors of shape (# channels) x height x width where height/width correspond to the
    ground plane of the environment.
    """

    def __init__(
        self,
        frame_height: int,
        frame_width: int,
        n_map_channels: int,
        resolution_in_cm: int = 5,
        map_size_in_cm: int = 2400,
        vision_range_in_cm: int = 300,
        use_pose_estimation: bool = False,
        pretrained_resnet: bool = True,
        freeze_resnet_batchnorm: bool = True,
        use_resnet_layernorm: bool = False,
    ):
        """Initialize an Active Neural SLAM module.

        # Parameters

        frame_height : The height of the RGB images given to this module on calls to `forward`.
        frame_width : The width of the RGB images given to this module on calls to `forward`.
        n_map_channels : The number of output channels in the output maps.
        resolution_in_cm : The resolution of the output map, see `map_size_in_cm`.
        map_size_in_cm : The height & width of the map in centimeters. The size of the map
            tensor returned on calls to forward will be `map_size_in_cm/resolution_in_cm`. Note
            that `map_size_in_cm` must be an divisible by resolution_in_cm.
        vision_range_in_cm : Given an RGB image input, this module will transform this image into
            an "egocentric map" with height and width equaling `vision_range_in_cm/resolution_in_cm`.
            This egocentr map corresponds to the area of the world directly in front of the agent.
            This "egocentric map" will be rotated/translated into the allocentric reference frame and
            used to update the larger, allocentric, map whose
            height and width equal `map_size_in_cm/resolution_in_cm`. Thus this parameter controls
            how much of the map will be updated on every step.
        use_pose_estimation : Whether or not we should estimate the agent's change in position/rotation.
            If `False`, you'll need to provide the ground truth changes in position/rotation.
        pretrained_resnet : Whether or not to use ImageNet pre-trained model weights for the ResNet18
            backbone.
        freeze_resnet_batchnorm : Whether or not the batch normalization layers in the ResNet18 backbone
            should be frozen and batchnorm updates disabled. You almost certainly want this to be `True`
            as using batch normalization during RL training results in all sorts of issues unless you're
            very careful.
        use_resnet_layernorm : If you've enabled `freeze_resnet_batchnorm` (recommended) you'll likely want
            to normalize the output from the ResNet18 model as we've found that these values can otherwise
            grow quite large harming learning.
        """
        super(ActiveNeuralSLAM, self).__init__()
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.n_map_channels = n_map_channels
        self.resolution_in_cm = resolution_in_cm
        self.map_size_in_cm = map_size_in_cm
        self.input_channels = 3
        self.vision_range_in_cm = vision_range_in_cm
        self.dropout = 0.5
        self.use_pose_estimation = use_pose_estimation
        self.freeze_resnet_batchnorm = freeze_resnet_batchnorm

        self.max_abs_map_logit_value = 20

        # Visual Encoding
        resnet = models.resnet18(pretrained=pretrained_resnet)
        self.resnet_l5 = nn.Sequential(*list(resnet.children())[0:8])
        self.conv = nn.Sequential(
            *filter(bool, [nn.Conv2d(512, 64, (1, 1), stride=(1, 1)), nn.ReLU()])
        )
        self.bn_modules = [
            module
            for module in self.resnet_l5.modules()
            if "BatchNorm" in type(module).__name__
        ]
        if freeze_resnet_batchnorm:
            for bn in self.bn_modules:
                bn.momentum = 0

        # Layernorm (if requested)
        self.use_resnet_layernorm = use_resnet_layernorm
        if self.use_resnet_layernorm:
            assert (
                self.freeze_resnet_batchnorm
            ), "When using layernorm, we require that set `freeze_resnet_batchnorm` to True."
            self.resnet_normalizer = nn.Sequential(
                nn.Conv2d(512, 512, 1),
                nn.LayerNorm(
                    normalized_shape=[512, 7, 7],
                    elementwise_affine=True,
                ),
            )
            self.resnet_normalizer.apply(simple_conv_and_linear_weights_init)
        else:
            self.resnet_normalizer = nn.Identity()

        # convolution output size
        input_test = torch.randn(
            1, self.input_channels, self.frame_height, self.frame_width
        )
        # Have to explicitly call .forward to get past LGTM checks as it thinks nn.Sequential isn't callable
        conv_output = self.conv.forward(self.resnet_l5.forward(input_test))

        self.conv_output_size = conv_output.view(-1).size(0)

        # projection layer
        self.proj1 = nn.Linear(self.conv_output_size, 1024)
        assert self.vision_range % 8 == 0
        self.deconv_in_height = self.vision_range // 8
        self.deconv_in_width = self.deconv_in_height
        self.n_input_channels_for_deconv = 64
        proj2_out_size = 64 * self.deconv_in_height * self.deconv_in_width
        self.proj2 = nn.Linear(1024, proj2_out_size)

        if self.dropout > 0:
            self.dropout1 = nn.Dropout(self.dropout)
            self.dropout2 = nn.Dropout(self.dropout)

        # Deconv layers to predict map
        self.deconv = nn.Sequential(
            *filter(
                bool,
                [
                    nn.ConvTranspose2d(
                        self.n_input_channels_for_deconv,
                        32,
                        (4, 4),
                        stride=(2, 2),
                        padding=(1, 1),
                    ),
                    nn.ReLU(),
                    nn.ConvTranspose2d(32, 16, (4, 4), stride=(2, 2), padding=(1, 1)),
                    nn.ReLU(),
                    nn.ConvTranspose2d(
                        16, self.n_map_channels, (4, 4), stride=(2, 2), padding=(1, 1)
                    ),
                ],
            )
        )

        # Pose Estimator
        self.pose_conv = nn.Sequential(
            nn.Conv2d(2 * self.n_map_channels, 64, (4, 4), stride=(2, 2)),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, (4, 4), stride=(2, 2)),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, (3, 3), stride=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Flatten(),
        )

        self.pose_conv_output_dim = (
            self.pose_conv.forward(
                torch.zeros(
                    1, 2 * self.n_map_channels, self.vision_range, self.vision_range
                )
            )
            .view(-1)
            .size(0)
        )

        # projection layer
        self.pose_proj1 = nn.Linear(self.pose_conv_output_dim, 1024)
        self.pose_proj2_x = nn.Linear(1024, 128)
        self.pose_proj2_z = nn.Linear(1024, 128)
        self.pose_proj2_o = nn.Linear(1024, 128)
        self.pose_proj3_x = nn.Linear(128, 1)
        self.pose_proj3_y = nn.Linear(128, 1)
        self.pose_proj3_o = nn.Linear(128, 1)

        if self.dropout > 0:
            self.pose_dropout1 = nn.Dropout(self.dropout)

        self.train()

    @property
    def device(self):
        d = self.pose_proj1.weight.get_device()
        if d < 0:
            return torch.device("cpu")
        return torch.device(d)

    def train(self, mode: bool = True):
        super().train(mode=mode)
        if mode and self.freeze_resnet_batchnorm:
            for module in self.bn_modules:
                module.eval()

    @property
    def map_size(self):
        return self.map_size_in_cm // self.resolution_in_cm

    @property
    def vision_range(self):
        return self.vision_range_in_cm // (self.resolution_in_cm)

    def image_to_egocentric_map_logits(
        self,
        images: Optional[torch.Tensor],
        resnet_image_features: Optional[torch.Tensor] = None,
    ):
        if resnet_image_features is None:
            bs, _, _, _ = images.size()
            resnet_image_features = self.resnet_normalizer(
                self.resnet_l5(images[:, :3, :, :])
            )
        else:
            bs = resnet_image_features.shape[0]

        conv_output = self.conv(resnet_image_features)

        proj1 = F.relu(self.proj1(conv_output.reshape(-1, self.conv_output_size)))
        if self.dropout > 0:
            proj1 = self.dropout1(proj1)
        proj3 = F.relu(self.proj2(proj1))

        deconv_input = proj3.view(
            bs,
            self.n_input_channels_for_deconv,
            self.deconv_in_height,
            self.deconv_in_width,
        )
        deconv_output = self.deconv(deconv_input)
        return deconv_output

    def allocentric_map_to_egocentric_view(
        self, allocentric_map: torch.Tensor, xzr: torch.Tensor, padding_mode: str
    ):
        # Index the egocentric viewpoints at the given xzr locations
        with torch.no_grad():
            allocentric_map = allocentric_map.float()
            xzr = xzr.float()

            theta = xzr[:, 2].float() * float(np.pi / 180)

            # Here form the rotation matrix
            cos_theta = torch.cos(theta)
            sin_theta = torch.sin(theta)
            rot_mat = torch.stack(
                (
                    torch.stack((cos_theta, -sin_theta), -1),
                    torch.stack((sin_theta, cos_theta), -1),
                ),
                1,
            )

            scaler = 2 * (100 / (self.resolution_in_cm * self.map_size))
            offset_to_center_the_agent = scaler * xzr[:, :2].unsqueeze(-1) - 1

            offset_to_top_of_image = rot_mat @ torch.FloatTensor([0, 1.0]).unsqueeze(
                1
            ).to(self.device)
            rotation_and_translate_mat = torch.cat(
                (
                    rot_mat,
                    offset_to_top_of_image + offset_to_center_the_agent,
                ),
                dim=-1,
            )

            ego_map = F.grid_sample(
                allocentric_map,
                F.affine_grid(
                    rotation_and_translate_mat.to(self.device),
                    allocentric_map.shape,
                ),
                padding_mode=padding_mode,
                align_corners=False,
            )

            vr = self.vision_range
            half_vr = vr // 2
            center = self.map_size_in_cm // (2 * self.resolution_in_cm)
            cropped = ego_map[:, :, :vr, (center - half_vr) : (center + half_vr)]
            return cropped

    def estimate_egocentric_dx_dz_dr(
        self,
        map_probs_egocentric: torch.Tensor,
        last_map_probs_egocentric: torch.Tensor,
    ):
        assert last_map_probs_egocentric.shape == map_probs_egocentric.shape

        pose_est_input = torch.cat(
            (map_probs_egocentric.detach(), last_map_probs_egocentric.detach()), dim=1
        )
        pose_conv_output = self.pose_conv(pose_est_input)

        proj1 = F.relu(self.pose_proj1(pose_conv_output))

        if self.dropout > 0:
            proj1 = self.pose_dropout1(proj1)

        proj2_x = F.relu(self.pose_proj2_x(proj1))
        pred_dx = self.pose_proj3_x(proj2_x)

        proj2_z = F.relu(self.pose_proj2_z(proj1))
        pred_dz = self.pose_proj3_y(proj2_z)

        proj2_o = F.relu(self.pose_proj2_o(proj1))
        pred_do = self.pose_proj3_o(proj2_o)

        return torch.cat((pred_dx, pred_dz, pred_do), dim=1)

    @staticmethod
    def update_allocentric_xzrs_with_egocentric_movement(
        last_xzrs_allocentric: torch.Tensor,
        dx_dz_drs_egocentric: torch.Tensor,
    ):
        new_xzrs_allocentric = last_xzrs_allocentric.clone()

        theta = new_xzrs_allocentric[:, 2] * DEGREES_TO_RADIANS
        sin_theta = torch.sin(theta)
        cos_theta = torch.cos(theta)
        new_xzrs_allocentric[:, :2] += torch.matmul(
            torch.stack([cos_theta, -sin_theta, sin_theta, cos_theta], dim=-1).view(
                -1, 2, 2
            ),
            dx_dz_drs_egocentric[:, :2].unsqueeze(-1),
        ).squeeze(-1)

        new_xzrs_allocentric[:, 2] += dx_dz_drs_egocentric[:, 2]
        new_xzrs_allocentric[:, 2] = (
            torch.fmod(new_xzrs_allocentric[:, 2] - 180.0, 360.0) + 180.0
        )
        new_xzrs_allocentric[:, 2] = (
            torch.fmod(new_xzrs_allocentric[:, 2] + 180.0, 360.0) - 180.0
        )

        return new_xzrs_allocentric

    def forward(
        self,
        images: Optional[torch.Tensor],
        last_map_probs_allocentric: Optional[torch.Tensor],
        last_xzrs_allocentric: Optional[torch.Tensor],
        dx_dz_drs_egocentric: Optional[torch.Tensor],
        last_map_logits_egocentric: Optional[torch.Tensor],
        return_allocentric_maps=True,
        resnet_image_features: Optional[torch.Tensor] = None,
    ):
        """Create allocentric/egocentric maps predictions given RGB image
        inputs.

        Here it is assumed that `last_xzrs_allocentric` has been re-centered so that (x, z) == (0,0)
        corresponds to the top left of the returned map (with increasing x/z moving to the bottom right of the map).

        Note that all maps are oriented so that:
        * **Increasing x values** correspond to **increasing columns** in the map(s).
        * **Increasing z values** correspond to **increasing rows** in the map(s).
        Note that this may seem a bit weird as:
        * "north" is pointing downwards in the map,
        * if you picture yourself as the agent facing north (i.e. down) in the map, then moving to the right from
            the agent's perspective will correspond to **increasing** which column the agent is at:
        ```
        agent facing downwards - - > (dir. to the right of the agent, i.e. moving right corresponds to +cols)
            |
            |
            v (dir. agent faces, i.e. moving ahead corresponds to +rows)
        ```
            This may be the opposite of what you expect.

        # Parameters
        images : A (# batches) x 3 x height x width tensor of RGB images. These should be
            normalized for use with a resnet model. See [here](https://pytorch.org/vision/stable/models.html)
            for information (see also the `use_resnet_normalization` parameter of the
            `allenact.base_abstractions.sensor.RGBSensor` sensor).
        last_map_probs_allocentric : A (# batches) x (map channels) x (map height) x (map width)
            tensor representing the colllection of allocentric maps to be updated.
        last_xzrs_allocentric : A (# batches) x 3 tensor where `last_xzrs_allocentric[:, 0]`
            are the agent's (allocentric) x-coordinates on the previous step,
            `last_xzrs_allocentric[:, 1]` are the agent's (allocentric) z-coordinates from the previous
            step, and `last_xzrs_allocentric[:, 2]` are the agent's rotations (allocentric, in degrees)
            from the prevoius step.
        dx_dz_drs_egocentric : A (# batches) x 3 tensor representing the agent's change in x (in meters), z (in meters), and rotation (in degrees)
            from the previous step. Note that these changes are "egocentric" so that if the agent moved
            1 meter ahead from it's perspective this should correspond to a dz of +1.0 regardless of
            the agent's orientation (similarly moving right would result in a dx of +1.0). This
            is ignored (and thus can be `None`) if you are using pose estimation
            (i.e. `self.use_pose_estimation` is `True`) or if `return_allocentric_maps` is `False`.
        last_map_logits_egocentric : The "egocentric_update" output when calling this function
            on the last agent's step. I.e. this should be the egocentric map view of the agent
            from the last step. This is used to compute the change in the agent's position rotation.
            This is ignored (and thus can be `None`) if you do not wish to estimate the agent's pose
            (i.e. `self.use_pose_estimation` is `False`).
        return_allocentric_maps : Whether or not to generate new allocentric maps given `last_map_probs_allocentric`
            and the new map estimates. Creating these new allocentric maps is expensive so better avoided when
            not needed.
        resnet_image_features : Sometimes you may wish to compute the ResNet image features yourself for use
            in another part of your model. Rather than having to recompute them multiple times, you can
            instead compute them once and pass them into this forward call (in this case the input `images`
            parameter is ignored). Note that if you're using the `self.resnet_l5` module to compute these
            features, be sure to also normalize them with `self.resnet_normalizer` if you have opted to
            `use_resnet_layernorm` when initializing this module).

        # Returns
        A dictionary with keys/values:
        * "egocentric_update" - The egocentric map view for the given RGB image. This is what should
            be used for computing losses in general.
        * "map_logits_probs_update_no_grad" - The egocentric map view after it has been
            rotated, translated, and moved into a full-sized allocentric map. This map has been
            detached from the computation graph and so should not be used for gradient computations.
            This will be `None` if `return_allocentric_maps` was `False`.
        * "map_logits_probs_no_grad" - The newly updated allocentric map, this corresponds to
            performing a pointwise maximum between `last_map_probs_allocentric` and the
            above returned `map_probs_allocentric_update_no_grad`.
            This will be `None` if `return_allocentric_maps` was `False`.
        * "dx_dz_dr_egocentric_preds" - The predicted change in x, z, and rotation of the agent (from the
            egocentric perspective of the agent).
        *  "xzr_allocentric_preds" - The (predicted if `self.use_pose_estimation == True`) allocentric
            (x, z) position and rotation of the agent. This will equal `None` if `self.use_pose_estimation == False`
            and `dx_dz_drs_egocentric` is `None`.
        """
        # TODO: For consistency we should update things so that:
        #  "Furthermore, the rotation component of `last_xzrs_allocentric` and `dx_dz_drs_egocentric`
        #  should be specified in **degrees* with positive rotation corresponding to a **CLOCKWISE**
        #  rotation (this is the default used by the many game engines)."
        map_logits_egocentric = self.image_to_egocentric_map_logits(
            images=images, resnet_image_features=resnet_image_features
        )
        map_probs_egocentric = torch.sigmoid(map_logits_egocentric)

        dx_dz_dr_egocentric_preds = None
        if last_map_logits_egocentric is not None:
            dx_dz_dr_egocentric_preds = self.estimate_egocentric_dx_dz_dr(
                map_probs_egocentric=map_probs_egocentric,
                last_map_probs_egocentric=torch.sigmoid(last_map_logits_egocentric),
            )

        if self.use_pose_estimation:
            updated_xzrs_allocentrc = (
                self.update_allocentric_xzrs_with_egocentric_movement(
                    last_xzrs_allocentric=last_xzrs_allocentric,
                    dx_dz_drs_egocentric=dx_dz_dr_egocentric_preds,
                )
            )
        elif dx_dz_drs_egocentric is not None:
            updated_xzrs_allocentrc = (
                self.update_allocentric_xzrs_with_egocentric_movement(
                    last_xzrs_allocentric=last_xzrs_allocentric,
                    dx_dz_drs_egocentric=dx_dz_drs_egocentric,
                )
            )
        else:
            updated_xzrs_allocentrc = None

        if return_allocentric_maps:
            # Aggregate egocentric map prediction in the allocentric map
            # using the predicted pose (if `self.use_pose_estimation`) or the ground
            # truth pose (if not `self.use_pose_estimation`)
            with torch.no_grad():
                # Rotate and translate the egocentric map view, we do this grid sampling
                # at the level of probabilities as bad results can occur at the logit level
                full_size_allocentric_map_probs_update = (
                    _move_egocentric_map_view_into_allocentric_position(
                        map_probs_egocentric=map_probs_egocentric,
                        xzrs_allocentric=updated_xzrs_allocentrc,
                        allocentric_map_height_width=(self.map_size, self.map_size),
                        resolution_in_cm=self.resolution_in_cm,
                    )
                )

                map_probs_allocentric = torch.max(
                    last_map_probs_allocentric, full_size_allocentric_map_probs_update
                )
        else:
            full_size_allocentric_map_probs_update = None
            map_probs_allocentric = None

        return {
            "egocentric_update": map_logits_egocentric,
            "map_probs_allocentric_update_no_grad": full_size_allocentric_map_probs_update,
            "map_probs_allocentric_no_grad": map_probs_allocentric,
            "dx_dz_dr_egocentric_preds": dx_dz_dr_egocentric_preds,
            "xzr_allocentric_preds": updated_xzrs_allocentrc,
        }


def _move_egocentric_map_view_into_allocentric_position(
    map_probs_egocentric: torch.Tensor,
    xzrs_allocentric: torch.Tensor,
    allocentric_map_height_width: Tuple[int, int],
    resolution_in_cm: float,
):
    """Translate/rotate an egocentric map view into an allocentric map.

    Let's say you have a collection of egocentric maps in a tensor of shape
    `(# batches) x (# channels) x (# ego rows) x (# ego columns)`
    where these are "egocentric" as we assume the agent is always
    at the center of the map and facing "downwards", namely
    * **ahead** of the agent should correspond to **increasing rows** in the map(s).
    * **right** of the agent should correspond to **increasing columns** in the map(s).
    Note that the above is a bit weird as, if you picture yourself as the agent facing
    downwards in the map, then moving to the right from the agent perspective. Here's how things
    should look if you plotted one of these egocentric maps:
    ```
    center of map - - > (dir. to the right of the agent, i.e. moving right corresponds to +cols)
        |
        |
        v (dir. agent faces, i.e. moving ahead corresponds to +rows)
    ```

    This function is used to translate/rotate the above ego maps so that
    they are in the right position/rotation in an allocentric map of size
    `(# batches) x (# channels) x (# allocentric_map_height_width[0]) x (# allocentric_map_height_width[1])`.

    Adapted from the get_grid function in https://github.com/devendrachaplot/Neural-SLAM.

    # Parameters
    map_probs_egocentric : Egocentric map views.
    xzrs_allocentric : (# batches)x3 tensor with `xzrs_allocentric[:, 0]` being the x-coordinates (in meters),
        `xzrs_allocentric[:, 1]` being the z-coordinates (in meters), and `xzrs_allocentric[:, 2]` being the rotation
        (in degrees) of the agent in the allocentric reference frame. Here it is assumed that `xzrs_allocentric` has
        been re-centered so that (x, z) == (0,0) corresponds to the top left of the returned map (with increasing
        x/z moving to the bottom right of the map). Note that positive rotations are in the counterclockwise direction.
    allocentric_map_height_width : Height/width of the allocentric map to be returned
    resolution_in_cm : Resolution (in cm) of map to be returned (and of map_probs_egocentric). I.e.
        `map_probs_egocentric[0,0,0:1,0:1]` should correspond to a `resolution_in_cm x resolution_in_cm`
        square on the ground plane in the world.

    # Returns
    `(# batches) x (# channels) x (# allocentric_map_height_width[0]) x (# allocentric_map_height_width[1])`
    tensor where the input `map_probs_egocentric` maps have been rotated/translated so that they
    are in the positions specified by `xzrs_allocentric`.
    """
    # TODO: For consistency we should update the rotations so they are in the clockwise direction.

    # First we place the egocentric map view into the center
    # of a map that has the same size as the allocentric map

    nbatch, c, ego_h, ego_w = map_probs_egocentric.shape
    allo_h, allo_w = allocentric_map_height_width

    max_view_range = math.sqrt((ego_w / 2.0) ** 2 + ego_h ** 2)
    if min(allo_h, allo_w) / 2.0 < max_view_range:
        raise NotImplementedError(
            f"The shape of your egocentric view (ego_h, ego_w)==({ego_h, ego_w})"
            f" is too large relative the size of the allocentric map (allo_h, allo_w)==({allo_h}, {allo_w})."
            f" The height/width of your allocentric map should be at least {2 * max_view_range} to allow"
            f" for no information to be lost when rotating the egocentric map."
        )

    full_size_ego_map_update_probs = map_probs_egocentric.new(
        nbatch, c, *allocentric_map_height_width
    ).fill_(0)

    assert (ego_h % 2, ego_w % 2, allo_h % 2, allo_w % 2) == (
        0,
    ) * 4, "All map heights/widths should be divisible by 2."

    x1 = allo_w // 2 - ego_w // 2
    x2 = x1 + ego_w
    z1 = allo_h // 2
    z2 = z1 + ego_h
    full_size_ego_map_update_probs[:, :, z1:z2, x1:x2] = map_probs_egocentric

    # Now we'll rotate and translate `full_size_ego_map_update_probs`
    # so that the egocentric map view is positioned where it should be
    # in the allocentric coordinate frame

    # To do this we first need to rescale our allocentric xz coordinates
    # so that the center of the map is (0,0) and the top left corner is (-1, -1)
    # as this is what's expected by the `affine_grid` function below.
    rescaled_xzrs_allocentric = xzrs_allocentric.clone().detach().float()
    rescaled_xzrs_allocentric[:, :2] *= (
        100.0 / resolution_in_cm
    )  # Put x / z into map units rather than meters
    rescaled_xzrs_allocentric[:, 0] /= allo_w / 2  # x corresponds to columns
    rescaled_xzrs_allocentric[:, 1] /= allo_h / 2  # z corresponds to rows
    rescaled_xzrs_allocentric[:, :2] -= 1.0  # Re-center

    x = rescaled_xzrs_allocentric[:, 0]
    z = rescaled_xzrs_allocentric[:, 1]
    theta = (
        -rescaled_xzrs_allocentric[:, 2] * DEGREES_TO_RADIANS
    )  # Notice the negative sign

    cos_theta = theta.cos()
    sin_theta = theta.sin()
    zeroes = torch.zeros_like(cos_theta)
    ones = torch.ones_like(cos_theta)

    theta11 = torch.stack([cos_theta, -sin_theta, zeroes], 1)
    theta12 = torch.stack([sin_theta, cos_theta, zeroes], 1)
    theta1 = torch.stack([theta11, theta12], 1)

    theta21 = torch.stack([ones, zeroes, x], 1)
    theta22 = torch.stack([zeroes, ones, z], 1)
    theta2 = torch.stack([theta21, theta22], 1)

    grid_size = (nbatch, c, allo_h, allo_w)
    rot_grid = F.affine_grid(theta1, grid_size)
    trans_grid = F.affine_grid(theta2, grid_size)

    return F.grid_sample(
        F.grid_sample(
            full_size_ego_map_update_probs,
            rot_grid,
            padding_mode="zeros",
            align_corners=False,
        ),
        trans_grid,
        padding_mode="zeros",
        align_corners=False,
    )
