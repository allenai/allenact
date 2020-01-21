import typing
from typing import Dict, Any, Callable, List

import torch
from torchvision import models
import numpy as np
import gym

from rl_base.preprocessor import Preprocessor


class ResNetEmbedder:
    def __init__(self, resnet, pool=True):
        self.model = resnet
        self.pool = pool
        self.model.eval()

    def __call__(self, x):
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
    def __init__(self, config: Dict[str, Any], *args: Any, **kwargs: Any):
        super().__init__(config, *args, **kwargs)

        def f(x, k, default):
            return x[k] if k in x else default

        self.input_height: int = f(config, "input_height", None)
        self.input_width: int = f(config, "input_width", None)
        self.output_height: int = f(config, "output_height", None)
        self.output_width: int = f(config, "output_width", None)
        self.output_dims: int = f(config, "output_dims", None)
        self.model: Callable = f(config, "torchvision_resnet_model", models.resnet18)
        self.device: str = f(config, "device", "cpu")

        assert (
            (self.input_height is not None)
            and (self.input_width is not None)
            and (self.output_height is not None)
            and (self.output_width is not None)
            and (self.output_dims is not None)
        ), (
            "In ResnetSensorThor's config, "
            "input and output heights and widths and output dims must be set."
        )

        self.resnet = ResNetEmbedder(
            self.model(pretrained=True).to(self.device), pool=True
        )

        low = -np.inf
        high = np.inf
        shape = (self.output_dims, self.output_height, self.output_width)
        self.observation_space = gym.spaces.Box(low=low, high=high, shape=shape)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return "resnet_thor"

    def _get_input_uuids(self, *args: Any, **kwargs: Any) -> List[str]:
        assert (
            len(self.config["input_uuids"]) == 1
        ), "preprocessor can only consume one observation"
        return self.config["input_uuids"]

    def _get_observation_space(self) -> gym.spaces.Box:
        return typing.cast(gym.spaces.Box, self.observation_space)

    def process(self, obs: Dict[str, Any], *args: Any, **kwargs: Any) -> Any:
        x = obs[self.input_uuids[0]].to(self.device).permute(0, 3, 1, 2)
        return self.resnet(x.to(self.device))
