import typing
from typing import Dict, Any, Callable, List

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


class ResnetPreProcessorThor(Preprocessor):
    """Preprocess RGB image using a ResNet model."""

    def __init__(self, config: Dict[str, Any], *args: Any, **kwargs: Any):
        super().__init__(config, *args, **kwargs)

        def f(x, k):
            assert k in x, "{} must be set in ResnetPreProcessorThor".format(k)
            return x[k]

        def optf(x, k, default):
            return x[k] if k in x else default

        self.input_height: int = f(config, "input_height")
        self.input_width: int = f(config, "input_width")
        self.output_height: int = f(config, "output_height")
        self.output_width: int = f(config, "output_width")
        self.output_dims: int = f(config, "output_dims")
        self.make_model: Callable[..., models.ResNet] = optf(
            config, "torchvision_resnet_model", models.resnet18
        )
        self.device: torch.device = optf(config, "device", "cpu")

        self.resnet = ResNetEmbedder(
            self.make_model(pretrained=True).to(self.device), pool=False
        )

        low = -np.inf
        high = np.inf
        shape = (self.output_dims, self.output_height, self.output_width)
        self.observation_space = gym.spaces.Box(low=low, high=high, shape=shape)

        assert (
            len(self.config["input_uuids"]) == 1
        ), "preprocessor can only consume one observation"

    def to(self, device: torch.device) -> "ResnetPreProcessorThor":
        self.resnet = self.resnet.to(device)
        return self

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return "resnet_thor"

    def _get_input_uuids(self, *args: Any, **kwargs: Any) -> List[str]:
        return self.config["input_uuids"]

    def _get_observation_space(self) -> gym.spaces.Box:
        return typing.cast(gym.spaces.Box, self.observation_space)

    def process(self, obs: Dict[str, Any], *args: Any, **kwargs: Any) -> Any:
        x = obs[self.input_uuids[0]].to(self.device).permute(0, 3, 1, 2)
        return self.resnet(x.to(self.device))
