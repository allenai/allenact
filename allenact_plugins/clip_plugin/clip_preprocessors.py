from typing import List, Optional, Any, cast, Dict

import clip
import gym
import numpy as np
import torch
import torch.nn as nn
from clip.model import CLIP

from allenact.base_abstractions.preprocessor import Preprocessor
from allenact.utils.misc_utils import prepare_locals_for_super


class ClipResNetEmbedder(nn.Module):
    def __init__(self, resnet: CLIP, pool=True, pooling_type="avg"):
        super().__init__()
        self.model = resnet
        self.pool = pool
        self.pooling_type = pooling_type

        if not pool:
            self.model.visual.attnpool = nn.Identity()
        elif self.pooling_type == "attn":
            pass
        elif self.pooling_type == "avg":
            self.model.visual.attnpool = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(start_dim=-3, end_dim=-1)
            )
        else:
            raise NotImplementedError("`pooling_type` must be 'avg' or 'attn'.")

        self.eval()

    def forward(self, x):
        with torch.no_grad():
            return self.model.visual(x)


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
        if self._resnet is None:
            self._resnet = ClipResNetEmbedder(
                clip.load(self.clip_model_type, device=self.device)[0], pool=self.pool
            ).to(self.device)
            for module in self._resnet.modules():
                if "BatchNorm" in type(module).__name__:
                    module.momentum = 0.0
            self._resnet.eval()
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
