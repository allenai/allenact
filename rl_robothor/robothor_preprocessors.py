import typing
from typing import Dict, Any, Optional, List, Union
from collections import OrderedDict
import logging

import gym
from gym.spaces.dict import Dict as SpaceDict
import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel

from utils.cacheless_frcnn import fasterrcnn_resnet50_fpn
from rl_base.preprocessor import Preprocessor

LOGGER = logging.getLogger("embodiedrl")


class BatchedFasterRCNN(torch.nn.Module):
    COCO_INSTANCE_CATEGORY_NAMES = [
        '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
        'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
        'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
        'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
        'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
        'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
        'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
        'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
        'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]

    def __init__(self, thres=0.12, maxdets=3, res=7):
        super().__init__()
        self.model = fasterrcnn_resnet50_fpn(pretrained=True)
        self.eval()

        self.min_score = thres
        self.maxdets = maxdets
        self.res = res

    def detector_tensor(self, boxes, classes, scores, aspect_ratio=1.0):
        res, maxdets = self.res, self.maxdets
        bins = np.array(list(range(res + 1)))[1:-1] / res

        res_classes = torch.zeros(res, res, maxdets, dtype=torch.int64)  # 0 is background
        res_boxes = -1 * torch.ones(res, res, maxdets, 5)  # regular range is [0, 1] (vert) or [0, aspect_ratio] (horiz)

        temp = [[[] for _ in range(res)] for _ in range(res)]  # grid of arrays

        # # TODO Debug
        # print('NEW IMAGE')

        for it in range(classes.shape[0]):
            cx = (boxes[it, 0].item() + boxes[it, 2].item()) / 2
            cy = (boxes[it, 1].item() + boxes[it, 3].item()) / 2

            px = np.digitize(cx, bins=aspect_ratio * bins).item()
            py = np.digitize(cy, bins=bins).item()

            temp[py][px].append((
                scores[it][classes[it]].item(),  # prob
                (boxes[it, 2] - boxes[it, 0]).item() / aspect_ratio,  # width
                (boxes[it, 3] - boxes[it, 1]).item(),  # height
                boxes[it, 0].item() / aspect_ratio,  # x
                boxes[it, 1].item(),  # y
                classes[it].item()  # class
            ))

            # # TODO Debug:
            # print(self.COCO_INSTANCE_CATEGORY_NAMES[classes[it].item()])

        for py in range(res):
            for px in range(res):
                order = sorted(temp[py][px], reverse=True)[:maxdets]
                for it, data in enumerate(order):
                    res_classes[py, px, it] = data[-1]
                    res_boxes[py, px, it, :] = torch.tensor(list(data[:-1]))  # prob, size, top left

        res_classes = res_classes.permute(2, 0, 1).unsqueeze(0).contiguous()
        res_boxes = res_boxes.view(res, res, -1).permute(2, 0, 1).unsqueeze(0).contiguous()

        return res_classes, res_boxes

    def forward(self, imbatch):
        with torch.no_grad():
            imglist = [im_in.squeeze(0) for im_in in imbatch.split(split_size=1, dim=0)]

            # # TODO Debug
            # import cv2
            # for it, im_in in enumerate(imglist):
            #     cvim = 255.0 * im_in.to('cpu').permute(1, 2, 0).numpy()[:, :, ::-1]
            #     cv2.imwrite('test_highres{}.png'.format(it), cvim)

            preds = self.model(imglist)

            keeps = [pred['scores'] > self.min_score for pred in preds]  # already  after nms

            # [0, 1] for rows, [0, aspect_ratio] for cols (im_in is C x H x W), with all images of same size (batch)
            all_boxes = [pred['boxes'][keep] / imbatch.shape[-2] for pred, keep in zip(preds, keeps)]
            all_classes = [pred['labels'][keep] for pred, keep in zip(preds, keeps)]
            all_pred_scores = [pred['scores'][keep] for pred, keep in zip(preds, keeps)]

            # hack: fill in a full prob score (all classes, 0 score if undetected) for each box, for backwards compatibility
            all_scores = [
                torch.zeros(pred_scores.shape[0], 91, device=pred_scores.device) for pred_scores in all_pred_scores
            ]
            all_scores = [
                torch.where(torch.arange(91, device=pred_scores.device).unsqueeze(0) == merged_classes.unsqueeze(1),
                            pred_scores.unsqueeze(1), scores)
                for merged_classes, pred_scores, scores in zip(all_classes, all_pred_scores, all_scores)
            ]

            all_classes_boxes = [
                self.detector_tensor(boxes, classes, scores, aspect_ratio=imbatch.shape[-1] / imbatch.shape[-2])
                for boxes, classes, scores in zip(all_boxes, all_classes, all_scores)
            ]

            classes = torch.cat([classes_boxes[0] for classes_boxes in all_classes_boxes], dim=0).to(imbatch.device)
            boxes = torch.cat([classes_boxes[1] for classes_boxes in all_classes_boxes], dim=0).to(imbatch.device)

        return classes, boxes


class FasterRCNNPreProcessorRoboThor(Preprocessor):
    """Preprocess RGB image using a ResNet model."""
    COCO_INSTANCE_CATEGORY_NAMES = BatchedFasterRCNN.COCO_INSTANCE_CATEGORY_NAMES

    def __init__(self, config: Dict[str, Any], *args: Any, **kwargs: Any):
        def f(x, k):
            assert k in x, "{} must be set in ResnetPreProcessorThor".format(k)
            return x[k]

        def optf(x, k, default):
            return x[k] if k in x else default

        self.input_height: int = f(config, "input_height")
        self.input_width: int = f(config, "input_width")
        self.max_dets: int = f(config, "max_dets")
        self.detector_spatial_res: int = f(config, "detector_spatial_res")
        self.detector_thres: float = f(config, "detector_thres")
        self.parallel: bool = optf(config, "parallel", True)
        self.device: torch.device = torch.device(optf(config, "device", "cuda" if self.parallel and torch.cuda.is_available() else "cpu"))
        self.device_ids: Optional[List[Union[torch.device, int]]] = optf(config, "device_ids", list(range(torch.cuda.device_count())))

        self.frcnn = BatchedFasterRCNN(
            thres=self.detector_thres,
            maxdets=self.max_dets,
            res=self.detector_spatial_res,
        )

        if self.parallel:
            assert torch.cuda.is_available(), "attempt to parallelize detector without cuda"
            LOGGER.info("Distributing detector")
            self.frcnn = self.frcnn.to(torch.device("cuda"))

            # store = torch.distributed.TCPStore("localhost", 4712, 1, True)
            # torch.distributed.init_process_group(backend="nccl", store=store, rank=0, world_size=1)
            # self.model = DistributedDataParallel(self.frcnn, device_ids=self.device_ids)

            self.frcnn = torch.nn.DataParallel(self.frcnn, device_ids=self.device_ids)  #, output_device=torch.cuda.device_count() - 1)
            LOGGER.info("Detected {} devices".format(torch.cuda.device_count()))
            # images = torch.cat([(1. / (1. + it)) * torch.ones(1, 3, 300, 400) for it in range(24)], dim=0)
            # self.frcnn(images[:timages, ...])

            # LOGGER.info("Detector distributed")

        spaces: OrderedDict[str, gym.Space] = OrderedDict()
        shape = (self.max_dets, self.detector_spatial_res, self.detector_spatial_res)
        spaces['frcnn_classes'] = gym.spaces.Box(
            low=0,  # 0 is bg
            high=len(self.COCO_INSTANCE_CATEGORY_NAMES) - 1,
            shape=shape,
            dtype=np.int64
        )
        shape = (self.max_dets * 5, self.detector_spatial_res, self.detector_spatial_res)
        spaces['frcnn_boxes'] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=shape)
        self.observation_space = SpaceDict(spaces=spaces)

        assert (
            len(config["input_uuids"]) == 1
        ), "fasterrcnn preprocessor can only consume one observation type"

        f(config, "output_uuid")  # check before calling superclass

        super().__init__(config, *args, **kwargs)

    def to(self, device: torch.device) -> "FasterRCNNPreProcessorRoboThor":
        if not self.parallel:
            self.frcnn = self.frcnn.to(device)
            self.device = device
        return self

    def process(self, obs: Dict[str, Any], *args: Any, **kwargs: Any) -> Any:
        frames_tensor = obs[self.input_uuids[0]].to(self.device).permute(0, 3, 1, 2)  # bhwc -> bchw (unnormalized)
        classes, boxes = self.frcnn(frames_tensor)

        return {'frcnn_classes': classes, 'frcnn_boxes': boxes}
