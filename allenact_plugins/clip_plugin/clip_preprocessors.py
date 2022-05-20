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


class ClipViTEmbedder(nn.Module):
    def __init__(self, model: CLIP, class_emb_only: bool = False):
        super().__init__()
        self.model = model
        self.model.visual.transformer.resblocks = nn.Sequential(
            *list(self.model.visual.transformer.resblocks)[:-1]
        )
        self.class_emb_only = class_emb_only

        self.eval()

    def forward(self, x):
        m = self.model.visual
        with torch.no_grad():
            x = m.conv1(x)  # shape = [*, width, grid, grid]
            x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
            x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
            x = torch.cat(
                [
                    m.class_embedding.to(x.dtype)
                    + torch.zeros(
                        x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
                    ),
                    x,
                ],
                dim=1,
            )  # shape = [*, grid ** 2 + 1, width]
            x = x + m.positional_embedding.to(x.dtype)
            x = m.ln_pre(x)

            x = x.permute(1, 0, 2)  # NLD -> LND
            x = m.transformer(x)
            x = x.permute(1, 0, 2)  # LND -> NLD

            if self.class_emb_only:
                return x[:, 0, :]
            else:
                return x


class ClipViTPreprocessor(Preprocessor):
    """Preprocess RGB or depth image using a ResNet model with CLIP model
    weights."""

    CLIP_RGB_MEANS = (0.48145466, 0.4578275, 0.40821073)
    CLIP_RGB_STDS = (0.26862954, 0.26130258, 0.27577711)

    def __init__(
        self,
        rgb_input_uuid: str,
        clip_model_type: str,
        class_emb_only: bool,
        device: Optional[torch.device] = None,
        device_ids: Optional[List[torch.device]] = None,
        **kwargs: Any,
    ):
        assert clip_model_type in clip.available_models()

        if clip_model_type == "ViT-B/32":
            output_shape = (7 * 7 + 1, 768)
        elif clip_model_type == "ViT-B/16":
            output_shape = (14 * 14 + 1, 768)
        elif clip_model_type == "ViT-L/14":
            output_shape = (16 * 16 + 1, 1024)
        else:
            raise NotImplementedError(
                f"Currently `clip_model_type` must be one of 'ViT-B/32', 'ViT-B/16', or 'ViT-B/14'"
            )

        if class_emb_only:
            output_shape = output_shape[1:]

        self.clip_model_type = clip_model_type

        self.class_emb_only = class_emb_only

        self.device = torch.device("cpu") if device is None else device
        self.device_ids = device_ids or cast(
            List[torch.device], list(range(torch.cuda.device_count()))
        )
        self._vit: Optional[ClipViTEmbedder] = None

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
    def vit(self) -> ClipViTEmbedder:
        if self._vit is None:
            self._vit = ClipViTEmbedder(
                model=clip.load(self.clip_model_type, device=self.device)[0],
                class_emb_only=self.class_emb_only,
            ).to(self.device)
            for module in self._vit.modules():
                if "BatchNorm" in type(module).__name__:
                    module.momentum = 0.0
            self._vit.eval()
        return self._vit

    def to(self, device: torch.device) -> "ClipViTPreprocessor":
        self._vit = self.vit.to(device)
        self.device = device
        return self

    def process(self, obs: Dict[str, Any], *args: Any, **kwargs: Any) -> Any:
        x = obs[self.input_uuids[0]].to(self.device).permute(0, 3, 1, 2)  # bhwc -> bchw
        # If the input is depth, repeat it across all 3 channels
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        x = self.vit(x).float()
        return x
