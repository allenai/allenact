from typing import Any, Dict, Optional, List

import PIL
import gym
import numpy as np
from PIL import Image
from torchvision import transforms

from extensions.ai2thor.environment import AI2ThorEnvironment
from extensions.ai2thor.tasks import AI2ThorTask, ObjectNavTask
from rl_base.sensor import Sensor


class ScaleBothSides(object):
    """Rescales the input PIL.Image to the given 'width' and `height`.

    Attributes
        width: new width
        height: new height
        interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, width: int, height: int, interpolation=Image.BILINEAR):
        self.width = width
        self.height = height
        self.interpolation = interpolation

    def __call__(self, img: PIL.Image) -> PIL.Image:
        return img.resize((self.width, self.height), self.interpolation)


class RGBSensorThor(Sensor[AI2ThorEnvironment]):
    def __init__(self, config: Dict[str, Any], *args: Any, **kwargs: Any):
        super().__init__(config, *args, **kwargs)

        def f(x, k, default):
            return x[k] if k in x else default

        self.should_normalize = f(config, "use_resnet_normalization", False)
        self.height: Optional[int] = f(config, "height", None)
        self.width: Optional[int] = f(config, "width", None)
        self.should_normalize = f(config, "use_resnet_normalization", False)

        assert (self.width is None) == (self.height is None), (
            "In RGBSensorThor's config, "
            "either both height/width must be None or neither."
        )

        self.norm_means = np.array([[[0.485, 0.456, 0.406]]])
        self.norm_sds = np.array([[[0.229, 0.224, 0.225]]])

        shape = None if self.height is None else (self.height, self.width, 3)
        if not self.should_normalize:
            low = 0.0
            high = 1.0
        else:
            low = np.reshape(-self.norm_means / self.norm_sds, (3,))
            high = np.reshape((1 - self.norm_means) / self.norm_sds, (3,))
        self._observation_space = gym.spaces.Box(low=low, high=high, shape=shape)

        self.scaler = (
            None
            if self.width is None
            else ScaleBothSides(width=self.width, height=self.height)
        )

        self.to_pil = transforms.ToPILImage()

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return "rgb"

    def _get_observation_space(self) -> gym.spaces.Box:
        return self._observation_space

    def get_observation(
        self,
        env: AI2ThorEnvironment,
        task: Optional[AI2ThorTask],
        *args: Any,
        **kwargs: Any
    ) -> Any:
        rgb = env.current_frame
        assert rgb.dtype in [np.uint8, np.float32]

        if rgb.dtype is np.uint8:
            rgb = rgb / 255.0

        if self.should_normalize:
            rgb -= self.norm_means
            rgb /= self.norm_sds

        if self.scaler is not None and rgb.shape[:2] != (self.height, self.width):
            rgb = np.array(self.scaler(self.to_pil(rgb)), dtype=np.float32)

        return rgb


class GoalObjectTypeThorSensor(Sensor):
    def __init__(self, config: Dict[str, Any], *args: Any, **kwargs: Any):
        super().__init__(config, *args, **kwargs)

        self.ordered_object_types: List[str] = list(self.config["object_types"])
        assert self.ordered_object_types == list(sorted(self.ordered_object_types)), (
            "object types" "input to goal object type " "sensor must be ordered"
        )
        self.object_type_to_ind = {
            ot: i for i, ot in enumerate(self.ordered_object_types)
        }

        self._observation_space = gym.spaces.Discrete(len(self.ordered_object_types))

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return "goal_object_type_ind"

    def _get_observation_space(self) -> gym.spaces.Discrete:
        return self._observation_space

    def get_observation(
        self,
        env: AI2ThorEnvironment,
        task: Optional[ObjectNavTask],
        *args: Any,
        **kwargs: Any
    ) -> Any:
        return self.object_type_to_ind[task.task_info["object_type"]]
