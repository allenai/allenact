from typing import List, Callable, Optional, Any, cast, Dict

import gym
import numpy as np
import torch
import torch.nn as nn
from torchvision import models

import torch.nn.functional as F

from allenact.base_abstractions.preprocessor import Preprocessor
from allenact.utils.misc_utils import prepare_locals_for_super


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


class ResNetPreprocessor(Preprocessor):
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
        device: Optional[torch.device] = None,
        device_ids: Optional[List[torch.device]] = None,
        **kwargs: Any,
    ):
        def f(x, k):
            assert k in x, "{} must be set in ResNetPreprocessor".format(k)
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

        self.device = torch.device("cpu") if device is None else device
        self.device_ids = device_ids or cast(
            List[torch.device], list(range(torch.cuda.device_count()))
        )

        self._resnet: Optional[ResNetEmbedder] = None

        low = -np.inf
        high = np.inf
        shape = (self.output_dims, self.output_height, self.output_width)

        assert (
            len(input_uuids) == 1
        ), "resnet preprocessor can only consume one observation type"

        observation_space = gym.spaces.Box(low=low, high=high, shape=shape)

        super().__init__(**prepare_locals_for_super(locals()))

    @property
    def resnet(self) -> ResNetEmbedder:
        if self._resnet is None:
            self._resnet = ResNetEmbedder(
                self.make_model(pretrained=True).to(self.device), pool=self.pool
            )
        return self._resnet

    def to(self, device: torch.device) -> "ResNetPreprocessor":
        self._resnet = self.resnet.to(device)
        self.device = device
        return self

    def process(self, obs: Dict[str, Any], *args: Any, **kwargs: Any) -> Any:
        x = obs[self.input_uuids[0]].to(self.device).permute(0, 3, 1, 2)  # bhwc -> bchw
        # If the input is depth, repeat it across all 3 channels
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        return self.resnet(x.to(self.device))


class ClipResNetEmbedder(nn.Module):
    def __init__(self, resnet, pool=True):
        super().__init__()
        self.model = resnet
        self.pool = pool
        self.eval()

    def forward(self, x):
        m = self.model.visual
        with torch.no_grad():
            return m(x)


class ClipResNetPreprocessor(Preprocessor):
    """Preprocess RGB or depth image using a ResNet model with CLIP model
    weights."""

    CLIP_RGB_MEANS = (0.48145466, 0.4578275, 0.40821073)
    CLIP_RGB_STDS = (0.26862954, 0.26130258, 0.27577711)

    def __init__(
        self,
        rgb_input_uuid: str,
        clip_model_type: str,
        pool: bool,
        device: Optional[torch.device] = None,
        device_ids: Optional[List[torch.device]] = None,
        **kwargs: Any,
    ):
        try:
            import clip
        except ImportError as _:
            raise ImportError(
                "Cannot `import clip` when instatiating `CLIPResNetPreprocessor`."
                " Please install clip from the openai/CLIP git repository:"
                "\n`pip install git+https://github.com/openai/CLIP.git@3b473b0e682c091a9e53623eebc1ca1657385717`"
            )

        assert clip_model_type in clip.available_models()

        if clip_model_type == "RN50":
            output_shape = (2048, 7, 7)
        elif clip_model_type == "RN50x16":
            output_shape = (3072, 7, 7)
        else:
            raise NotImplementedError(
                f"Currently `clip_model_type` must be one of 'RN50' or 'RN50x16'"
            )

        if pool:
            output_shape = output_shape[:1]

        self.clip_model_type = clip_model_type

        self.pool = pool

        self.device = torch.device("cpu") if device is None else device
        self.device_ids = device_ids or cast(
            List[torch.device], list(range(torch.cuda.device_count()))
        )
        self._resnet: Optional[ClipResNetEmbedder] = None

        low = -np.inf
        high = np.inf
        shape = output_shape

        input_uuids = [rgb_input_uuid]
        assert (
            len(input_uuids) == 1
        ), "resnet preprocessor can only consume one observation type"

        observation_space = gym.spaces.Box(low=low, high=high, shape=shape)

        super().__init__(**prepare_locals_for_super(locals()))

    @property
    def resnet(self) -> ClipResNetEmbedder:
        import clip

        if self._resnet is None:
            self._resnet = ClipResNetEmbedder(
                clip.load(self.clip_model_type, device=self.device)[0], pool=self.pool
            ).to(self.device)
        return self._resnet

    def to(self, device: torch.device) -> "ClipResNetPreprocessor":
        self._resnet = self.resnet.to(device)
        self.device = device
        return self

    def process(self, obs: Dict[str, Any], *args: Any, **kwargs: Any) -> Any:
        x = obs[self.input_uuids[0]].to(self.device).permute(0, 3, 1, 2)  # bhwc -> bchw
        # If the input is depth, repeat it across all 3 channels
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        x = self.resnet(x).float()
        return x
