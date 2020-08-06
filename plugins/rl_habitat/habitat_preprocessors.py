from typing import Dict, Any, Callable, Optional, List, Union

import gym
import numpy as np
import torch
import torch.nn as nn
from torchvision import models

from rl_base.preprocessor import Preprocessor
from utils.misc_utils import prepare_locals_for_super
from utils.system import get_logger


class ResNetEmbedder(nn.Module):
    def __init__(self, resnet, pool=True):
        super().__init__()
        self.model = resnet
        self.pool = pool
        self.eval()

    def forward(self, x):
        with torch.no_grad():

            x = self.model.conv1(x)
            x = self.model.bn1(x)
            x = self.model.relu(x)
            x = self.model.maxpool(x)

            x = self.model.layer1(x)
            x = self.model.layer2(x)
            x = self.model.layer3(x)
            x = self.model.layer4(x)

            if not self.pool:
                return x
            else:
                x = self.model.avgpool(x)
                x = torch.flatten(x, 1)
                return x


class ResnetPreProcessorHabitat(Preprocessor):
    """Preprocess RGB or depth image using a ResNet model."""

    def __init__(
        self,
        input_uuids: List[str],
        output_uuid: str,
        input_height: int,
        input_width: int,
        output_height: int,
        output_width: int,
        output_dims: int,
        pool: bool,
        torchvision_resnet_model: Callable[..., models.ResNet] = models.resnet18,
        parallel: bool = True,
        device: Optional[torch.device] = None,
        device_ids: Optional[List[torch.device]] = None,
        **kwargs: Any
    ):
        def f(x, k):
            assert k in x, "{} must be set in ResnetPreProcessorHabitat".format(k)
            return x[k]

        def optf(x, k, default):
            return x[k] if k in x else default

        self.input_height = input_height
        self.input_width = input_width
        self.output_height = output_height
        self.output_width = output_width
        self.output_dims = output_dims
        self.pool = pool
        self.make_model = torchvision_resnet_model
        self.parallel = parallel
        self.device = (
            device
            if device is not None
            else ("cuda" if self.parallel and torch.cuda.is_available() else "cpu")
        )
        self.device_ids = device_ids or list(range(torch.cuda.device_count()))

        self.resnet: Union[
            ResNetEmbedder, torch.nn.DataParallel[ResNetEmbedder]
        ] = ResNetEmbedder(
            self.make_model(pretrained=True).to(self.device), pool=self.pool
        )

        if self.parallel:
            assert (
                torch.cuda.is_available()
            ), "attempt to parallelize resnet without cuda"
            get_logger().info("Distributing resnet")
            self.resnet = self.resnet.to(torch.device("cuda"))

            # store = torch.distributed.TCPStore("localhost", 4712, 1, True)
            # torch.distributed.init_process_group(backend="nccl", store=store, rank=0, world_size=1)
            # self.model = DistributedDataParallel(self.frcnn, device_ids=self.device_ids)

            self.resnet = torch.nn.DataParallel(
                self.resnet, device_ids=self.device_ids
            )  # , output_device=torch.cuda.device_count() - 1)
            get_logger().info("Detected {} devices".format(torch.cuda.device_count()))

        low = -np.inf
        high = np.inf
        shape = (self.output_dims, self.output_height, self.output_width)

        assert (
            len(input_uuids) == 1
        ), "resnet preprocessor can only consume one observation type"

        observation_space = gym.spaces.Box(low=low, high=high, shape=shape)

        super().__init__(**prepare_locals_for_super(locals()))

    def to(self, device: torch.device) -> "ResnetPreProcessorHabitat":
        if not self.parallel:
            self.resnet = self.resnet.to(device)
            self.device = device
        return self

    def process(self, obs: Dict[str, Any], *args: Any, **kwargs: Any) -> Any:
        x = obs[self.input_uuids[0]].to(self.device).permute(0, 3, 1, 2)  # bhwc -> bchw
        # If the input is depth, repeat it across all 3 channels
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        return self.resnet(x.to(self.device))
