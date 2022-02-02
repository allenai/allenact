# Original work Copyright (c) Facebook, Inc. and its affiliates.
# Modified work Copyright (c) Allen Institute for AI
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# Adapted from https://github.com/joel99/habitat-pointnav-aux/blob/master/habitat_baselines/

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gym.spaces.dict import Dict as SpaceDict

from allenact.utils.model_utils import Flatten
from allenact.utils.system import get_logger


def conv3x3(in_planes, out_planes, stride=1, groups=1):
    """3x3 convolution with padding."""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
        groups=groups,
    )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution."""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    resneXt = False

    def __init__(
        self, inplanes, planes, ngroups, stride=1, downsample=None, cardinality=1,
    ):
        super(BasicBlock, self).__init__()
        self.convs = nn.Sequential(
            conv3x3(inplanes, planes, stride, groups=cardinality),
            nn.GroupNorm(ngroups, planes),
            nn.ReLU(True),
            conv3x3(planes, planes, groups=cardinality),
            nn.GroupNorm(ngroups, planes),
        )
        self.downsample = downsample
        self.relu = nn.ReLU(True)

    def forward(self, x):
        residual = x

        out = self.convs(x)

        if self.downsample is not None:
            residual = self.downsample(x)

        return self.relu(out + residual)


def _build_bottleneck_branch(inplanes, planes, ngroups, stride, expansion, groups=1):
    return nn.Sequential(
        conv1x1(inplanes, planes),
        nn.GroupNorm(ngroups, planes),
        nn.ReLU(True),
        conv3x3(planes, planes, stride, groups=groups),
        nn.GroupNorm(ngroups, planes),
        nn.ReLU(True),
        conv1x1(planes, planes * expansion),
        nn.GroupNorm(ngroups, planes * expansion),
    )


class SE(nn.Module):
    def __init__(self, planes, r=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excite = nn.Sequential(
            nn.Linear(planes, int(planes / r)),
            nn.ReLU(True),
            nn.Linear(int(planes / r), planes),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        x = self.squeeze(x)
        x = x.view(b, c)
        x = self.excite(x)

        return x.view(b, c, 1, 1)


def _build_se_branch(planes, r=16):
    return SE(planes, r)


class Bottleneck(nn.Module):
    expansion = 4
    resneXt = False

    def __init__(
        self, inplanes, planes, ngroups, stride=1, downsample=None, cardinality=1,
    ):
        super().__init__()
        self.convs = _build_bottleneck_branch(
            inplanes, planes, ngroups, stride, self.expansion, groups=cardinality,
        )
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def _impl(self, x):
        identity = x

        out = self.convs(x)

        if self.downsample is not None:
            identity = self.downsample(x)

        return self.relu(out + identity)

    def forward(self, x):
        return self._impl(x)


class SEBottleneck(Bottleneck):
    def __init__(
        self, inplanes, planes, ngroups, stride=1, downsample=None, cardinality=1,
    ):
        super().__init__(inplanes, planes, ngroups, stride, downsample, cardinality)

        self.se = _build_se_branch(planes * self.expansion)

    def _impl(self, x):
        identity = x

        out = self.convs(x)
        out = self.se(out) * out

        if self.downsample is not None:
            identity = self.downsample(x)

        return self.relu(out + identity)


class SEResNeXtBottleneck(SEBottleneck):
    expansion = 2
    resneXt = True


class ResNeXtBottleneck(Bottleneck):
    expansion = 2
    resneXt = True


class GroupNormResNet(nn.Module):
    def __init__(self, in_channels, base_planes, ngroups, block, layers, cardinality=1):
        super(GroupNormResNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                base_planes,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            ),
            nn.GroupNorm(ngroups, base_planes),
            nn.ReLU(True),
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.cardinality = cardinality

        self.inplanes = base_planes
        if block.resneXt:
            base_planes *= 2

        self.layer1 = self._make_layer(block, ngroups, base_planes, layers[0])
        self.layer2 = self._make_layer(
            block, ngroups, base_planes * 2, layers[1], stride=2
        )
        self.layer3 = self._make_layer(
            block, ngroups, base_planes * 2 * 2, layers[2], stride=2
        )
        self.layer4 = self._make_layer(
            block, ngroups, base_planes * 2 * 2 * 2, layers[3], stride=2
        )

        self.final_channels = self.inplanes
        self.final_spatial_compress = 1.0 / (2 ** 5)

    def _make_layer(self, block, ngroups, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.GroupNorm(ngroups, planes * block.expansion),
            )

        layers = [
            block(
                self.inplanes,
                planes,
                ngroups,
                stride,
                downsample,
                cardinality=self.cardinality,
            )
        ]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, ngroups))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x


def gnresnet18(in_channels, base_planes, ngroups):
    model = GroupNormResNet(in_channels, base_planes, ngroups, BasicBlock, [2, 2, 2, 2])

    return model


def gnresnet50(in_channels, base_planes, ngroups):
    model = GroupNormResNet(in_channels, base_planes, ngroups, Bottleneck, [3, 4, 6, 3])

    return model


def gnresneXt50(in_channels, base_planes, ngroups):
    model = GroupNormResNet(
        in_channels,
        base_planes,
        ngroups,
        ResNeXtBottleneck,
        [3, 4, 6, 3],
        cardinality=int(base_planes / 2),
    )

    return model


def se_gnresnet50(in_channels, base_planes, ngroups):
    model = GroupNormResNet(
        in_channels, base_planes, ngroups, SEBottleneck, [3, 4, 6, 3]
    )

    return model


def se_gnresneXt50(in_channels, base_planes, ngroups):
    model = GroupNormResNet(
        in_channels,
        base_planes,
        ngroups,
        SEResNeXtBottleneck,
        [3, 4, 6, 3],
        cardinality=int(base_planes / 2),
    )

    return model


def se_gnresneXt101(in_channels, base_planes, ngroups):
    model = GroupNormResNet(
        in_channels,
        base_planes,
        ngroups,
        SEResNeXtBottleneck,
        [3, 4, 23, 3],
        cardinality=int(base_planes / 2),
    )

    return model


class GroupNormResNetEncoder(nn.Module):
    def __init__(
        self,
        observation_space: SpaceDict,
        rgb_uuid: Optional[str],
        depth_uuid: Optional[str],
        output_size: int,
        baseplanes=32,
        ngroups=32,
        make_backbone=None,
    ):
        super().__init__()

        self._inputs = []

        self.rgb_uuid = rgb_uuid
        if self.rgb_uuid is not None:
            assert self.rgb_uuid in observation_space.spaces
            self._n_input_rgb = observation_space.spaces[self.rgb_uuid].shape[2]
            assert self._n_input_rgb >= 0
            self._inputs.append(self.rgb_uuid)
        else:
            self._n_input_rgb = 0

        self.depth_uuid = depth_uuid
        if self.depth_uuid is not None:
            assert self.depth_uuid in observation_space.spaces
            self._n_input_depth = observation_space.spaces[self.depth_uuid].shape[2]
            assert self._n_input_depth >= 0
            self._inputs.append(self.depth_uuid)
        else:
            self._n_input_depth = 0

        if not self.is_blind:
            spatial_size = (
                observation_space.spaces[self._inputs[0]].shape[0] // 2
            )  # H (=W) / 2

            # RGBD into one model
            input_channels = self._n_input_rgb + self._n_input_depth  # C

            self.backbone = make_backbone(input_channels, baseplanes, ngroups)

            final_spatial = int(
                np.ceil(spatial_size * self.backbone.final_spatial_compress)
            )  # fix bug in habitat that uses int()
            after_compression_flat_size = 2048
            num_compression_channels = int(
                round(after_compression_flat_size / (final_spatial ** 2))
            )
            self.compression = nn.Sequential(
                nn.Conv2d(
                    self.backbone.final_channels,
                    num_compression_channels,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                ),
                nn.GroupNorm(1, num_compression_channels),
                nn.ReLU(True),
            )

            self.output_shape = (
                num_compression_channels,
                final_spatial,
                final_spatial,
            )

            self.head = nn.Sequential(
                Flatten(),
                nn.Linear(np.prod(self.output_shape), output_size),
                nn.ReLU(True),
            )

            self.layer_init()

    @property
    def is_blind(self):
        return self._n_input_rgb + self._n_input_depth == 0

    def layer_init(self):
        for layer in self.modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(layer.weight, nn.init.calculate_gain("relu"))
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, val=0)
        get_logger().debug("Initializing resnet encoder")

    def forward(self, observations):
        if self.is_blind:
            return None

        # TODO: the reshape follows compute_cnn_output()
        # but it's hard to make the forward as a nn.Module as cnn param
        nagents: Optional[int] = None
        nsteps: Optional[int] = None
        nsamplers: Optional[int] = None
        assert len(self._inputs) > 0

        cnn_input = []
        for mode in self._inputs:
            mode_obs = observations[mode]
            assert len(mode_obs.shape) in [
                5,
                6,
            ], "CNN input must have shape [STEP, SAMPLER, (AGENT,) dim1, dim2, dim3]"
            if len(mode_obs.shape) == 6:
                nsteps, nsamplers, nagents = mode_obs.shape[:3]
            else:
                nsteps, nsamplers = mode_obs.shape[:2]
            # Make FLAT_BATCH = nsteps * nsamplers (* nagents)
            mode_obs = mode_obs.view(
                (-1,) + mode_obs.shape[2 + int(nagents is not None) :]
            )
            # permute tensor to dimension [BATCH x CHANNEL x HEIGHT X WIDTH]
            mode_obs = mode_obs.permute(0, 3, 1, 2)
            cnn_input.append(mode_obs)

        x = torch.cat(cnn_input, dim=1)
        x = F.avg_pool2d(x, 2)  # 2x downsampling

        x = self.backbone(x)  # (256, 4, 4)
        x = self.compression(x)  # (128, 4, 4)
        x = self.head(x)  # (2048) -> (hidden_size)

        if nagents is not None:
            x = x.reshape((nsteps, nsamplers, nagents,) + x.shape[1:])
        else:
            x = x.reshape((nsteps, nsamplers,) + x.shape[1:])

        return x
