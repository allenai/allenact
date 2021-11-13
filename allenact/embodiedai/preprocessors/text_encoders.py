from typing import List, Optional, Any, cast, Dict

import gym
import numpy as np
import torch

from allenact.base_abstractions.preprocessor import Preprocessor
from allenact.utils.misc_utils import prepare_locals_for_super


class ClipTextPreprocessor(Preprocessor):

    def __init__(
        self,
        goal_sensor_uuid: str,
        object_types: List[str],
        device: Optional[torch.device] = None,
        device_ids: Optional[List[torch.device]] = None,
        **kwargs: Any,
    ):
        try:
            import clip
            self.clip = clip
        except ImportError as _:
            raise ImportError(
                "Cannot `import clip` when instatiating `CLIPResNetPreprocessor`."
                " Please install clip from the openai/CLIP git repository:"
                "\n`pip install git+https://github.com/openai/CLIP.git@3b473b0e682c091a9e53623eebc1ca1657385717`"
            )

        output_shape = (1024,)

        self.object_types = object_types

        self.device = torch.device("cpu") if device is None else device
        self.device_ids = device_ids or cast(
            List[torch.device], list(range(torch.cuda.device_count()))
        )

        low = -np.inf
        high = np.inf
        shape = output_shape

        observation_space = gym.spaces.Box(low=low, high=high, shape=shape)

        input_uuids = [goal_sensor_uuid]        

        super().__init__(**prepare_locals_for_super(locals()))

    @property
    def text_encoder(self):
        if self._clip_model is None:
            self._clip_model = self.clip.load('RN50', device=self.device)[0]
            self._clip_model.eval()
        return self._clip_model.encode_text

    def to(self, device: torch.device):
        self.device = device
        self._clip_model = None
        return self

    def process(self, obs: Dict[str, Any], *args: Any, **kwargs: Any) -> Any:
        object_inds = obs[self.input_uuids[0]]
        if len(object_inds.shape) == 1:
            object_inds = object_inds.unsqueeze(0)
        object_types = [self.object_types[ind] for ind in object_inds]
        x = self.clip.tokenize([f"navigate to the {obj}" for obj in object_types]).to(self.device)
        with torch.no_grad():
            return self.text_encoder(x).float()
