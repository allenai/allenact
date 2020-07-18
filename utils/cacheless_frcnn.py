from typing import List

import torch
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.faster_rcnn import FasterRCNN
from torchvision.models.detection.faster_rcnn import model_urls
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.utils import load_state_dict_from_url


class CachelessAnchorGenerator(AnchorGenerator):
    def forward(self, image_list, feature_maps):
        # type: (ImageList, List[Tensor])
        grid_sizes = list([feature_map.shape[-2:] for feature_map in feature_maps])
        image_size = image_list.tensors.shape[-2:]
        strides = [
            [int(image_size[0] / g[0]), int(image_size[1] / g[1])] for g in grid_sizes
        ]
        dtype, device = feature_maps[0].dtype, feature_maps[0].device
        self.set_cell_anchors(dtype, device)
        anchors_over_all_feature_maps = self.grid_anchors(grid_sizes, strides)
        anchors = torch.jit.annotate(List[List[torch.Tensor]], [])
        for i, (image_height, image_width) in enumerate(image_list.image_sizes):
            anchors_in_image = []
            for anchors_per_feature_map in anchors_over_all_feature_maps:
                anchors_in_image.append(anchors_per_feature_map)
            anchors.append(anchors_in_image)
        anchors = [torch.cat(anchors_per_image) for anchors_per_image in anchors]

        return anchors


def fasterrcnn_resnet50_fpn(
    pretrained=False, progress=True, num_classes=91, pretrained_backbone=True, **kwargs
):
    if pretrained:
        # no need to download the backbone if pretrained is set
        pretrained_backbone = False
    backbone = resnet_fpn_backbone("resnet50", pretrained_backbone)

    anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    rpn_anchor_generator = CachelessAnchorGenerator(anchor_sizes, aspect_ratios)
    model = FasterRCNN(
        backbone, num_classes, rpn_anchor_generator=rpn_anchor_generator, **kwargs
    )

    # min_size = 300
    # max_size = 400
    # anchor_sizes = ((12,), (24,), (48,), (96,), (192,))
    # aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    # rpn_anchor_generator = CachelessAnchorGenerator(
    #     anchor_sizes, aspect_ratios
    # )
    # model = FasterRCNN(backbone, num_classes, rpn_anchor_generator=rpn_anchor_generator, min_size=min_size, max_size=max_size, **kwargs)

    if pretrained:
        state_dict = load_state_dict_from_url(
            model_urls["fasterrcnn_resnet50_fpn_coco"], progress=progress
        )
        model.load_state_dict(state_dict)
    return model
