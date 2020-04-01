from typing import Dict, Any, Callable

import torch
import torch.nn as nn
from torchvision import models
import numpy as np
import gym

from rl_base.preprocessor import Preprocessor


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

    def __init__(self, config: Dict[str, Any], *args: Any, **kwargs: Any):
        def f(x, k):
            assert k in x, "{} must be set in ResnetPreProcessorHabitat".format(k)
            return x[k]

        def optf(x, k, default):
            return x[k] if k in x else default

        self.input_height: int = f(config, "input_height")
        self.input_width: int = f(config, "input_width")
        self.output_height: int = f(config, "output_height")
        self.output_width: int = f(config, "output_width")
        self.output_dims: int = f(config, "output_dims")
        self.pool: bool = f(config, "pool")
        self.make_model: Callable[..., models.ResNet] = optf(
            config, "torchvision_resnet_model", models.resnet18
        )
        self.device: torch.device = optf(config, "device", "cpu")

        self.resnet = ResNetEmbedder(
            self.make_model(pretrained=True).to(self.device), pool=self.pool
        )

        low = -np.inf
        high = np.inf
        shape = (self.output_dims, self.output_height, self.output_width)
        self.observation_space = gym.spaces.Box(low=low, high=high, shape=shape)

        assert (
            len(config["input_uuids"]) == 1
        ), "resnet preprocessor can only consume one observation type"

        f(config, "output_uuid")

        super().__init__(config, *args, **kwargs)

    def to(self, device: torch.device) -> "ResnetPreProcessorHabitat":
        self.resnet = self.resnet.to(device)
        self.device = device
        return self

    def process(self, obs: Dict[str, Any], *args: Any, **kwargs: Any) -> Any:
        x = obs[self.input_uuids[0]].to(self.device).permute(0, 3, 1, 2)
        # If the input is depth, repeat it across all 3 channels
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        return self.resnet(x.to(self.device))
