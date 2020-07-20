import typing
from typing import Any, Dict, Optional, List

import gym
import numpy as np
from torchvision import transforms

from rl_ai2thor.ai2thor_environment import AI2ThorEnvironment
from rl_ai2thor.object_nav.tasks import ObjectNavTask
from rl_base.sensor import Sensor
from rl_base.task import Task
from utils.tensor_utils import ScaleBothSides


class RGBSensorThor(Sensor[AI2ThorEnvironment, Task[AI2ThorEnvironment]]):
    """Sensor for RGB images in AI2-THOR.

    Returns from a running AI2ThorEnvironment instance, the current RGB
    frame corresponding to the agent's egocentric view.
    """

    def __init__(self, config: Dict[str, Any], *args: Any, **kwargs: Any):
        """Initializer.

        # Parameters

        config : If `config["use_resnet_normalization"]` is `True` then the RGB images from THOR will be normalized
            with means `[0.485, 0.456, 0.406]` and standard deviations `[0.229, 0.224, 0.225]` (i.e. using the standard
            resnet normalization). If both `config["height"]` and `config["width"]` are non-negative integers then
            the RGB image returned from AI2-THOR will be rescaled to have shape (config["height"], config["width"], 3)
            using bilinear sampling.
        args : Extra args. Currently unused.
        kwargs : Extra kwargs. Currently unused.
        """

        def f(x, k, default):
            return x[k] if k in x else default

        self._should_normalize = f(config, "use_resnet_normalization", False)
        self._height: Optional[int] = f(config, "height", None)
        self._width: Optional[int] = f(config, "width", None)
        self._uuid: str = f(config, "uuid", "rgb")

        assert (self._width is None) == (self._height is None), (
            "In RGBSensorThor's config, "
            "either both height/width must be None or neither."
        )

        self._norm_means = np.array([[[0.485, 0.456, 0.406]]], dtype=np.float32)
        self._norm_sds = np.array([[[0.229, 0.224, 0.225]]], dtype=np.float32)

        shape = None if self._height is None else (self._height, self._width, 3)
        if not self._should_normalize:
            low = 0.0
            high = 1.0
            self.observation_space = gym.spaces.Box(
                low=np.float32(low), high=np.float32(high), shape=shape
            )
        else:
            low = np.tile(-self._norm_means / self._norm_sds, shape[:-1] + (1,))
            high = np.tile((1 - self._norm_means) / self._norm_sds, shape[:-1] + (1,))
            self.observation_space = gym.spaces.Box(
                low=np.float32(low), high=np.float32(high)
            )

        self._scaler = (
            None
            if self._width is None
            else ScaleBothSides(
                width=typing.cast(int, self._width),
                height=typing.cast(int, self._height),
            )
        )

        self._to_pil = transforms.ToPILImage()

        super().__init__(
            config, *args, **kwargs
        )  # call it last so that user can assign a uuid

    @property
    def height(self) -> Optional[int]:
        """Height that RGB image will be rescale to have.

        # Returns

        The height as a non-negative integer or `None` if no rescaling is done.
        """
        return self._height

    @property
    def width(self) -> Optional[int]:
        """Width that RGB image will be rescale to have.

        # Returns

        The width as a non-negative integer or `None` if no rescaling is done.
        """
        return self._width

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self._uuid

    def _get_observation_space(self) -> gym.spaces.Box:
        return typing.cast(gym.spaces.Box, self.observation_space)

    def get_observation(
        self,
        env: AI2ThorEnvironment,
        task: Optional[Task[AI2ThorEnvironment]],
        *args: Any,
        **kwargs: Any
    ) -> Any:
        rgb = env.current_frame.copy()

        if self._scaler is not None and rgb.shape[:2] != (self._height, self._width):
            rgb = np.array(self._scaler(self._to_pil(rgb)), dtype=np.uint8)  # hwc

        assert rgb.dtype in [np.uint8, np.float32]

        if rgb.dtype == np.uint8:
            rgb = rgb.astype(np.float32) / 255.0

        if self._should_normalize:
            rgb -= self._norm_means
            rgb /= self._norm_sds

        return rgb


class GoalObjectTypeThorSensor(Sensor):
    def __init__(self, config: Dict[str, Any], *args: Any, **kwargs: Any):
        super().__init__(config, *args, **kwargs)

        self.ordered_object_types: List[str] = list(self.config["object_types"])
        assert self.ordered_object_types == list(sorted(self.ordered_object_types)), (
            "object types" "input to goal object type " "sensor must be ordered"
        )

        if "target_to_detector_map" not in self.config:
            self.object_type_to_ind = {
                ot: i for i, ot in enumerate(self.ordered_object_types)
            }

            self.observation_space = gym.spaces.Discrete(len(self.ordered_object_types))
        else:
            assert (
                "detector_types" in self.config
            ), "Missing detector_types for map {}".format(
                self.config["target_to_detector_map"]
            )
            self.target_to_detector = self.config["target_to_detector_map"]
            self.detector_types = self.config["detector_types"]

            detector_index = {ot: i for i, ot in enumerate(self.detector_types)}
            self.object_type_to_ind = {
                ot: detector_index[self.target_to_detector[ot]]
                for ot in self.ordered_object_types
            }

            self.observation_space = gym.spaces.Discrete(len(self.detector_types))

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return "goal_object_type_ind"

    def _get_observation_space(self) -> gym.spaces.Discrete:
        return typing.cast(gym.spaces.Discrete, self.observation_space)

    def get_observation(
        self,
        env: AI2ThorEnvironment,
        task: Optional[ObjectNavTask],
        *args: Any,
        **kwargs: Any
    ) -> Any:

        # # Debug
        # print(task.task_info["object_type"], '->',
        #       self.detector_types[self.object_type_to_ind[task.task_info["object_type"]]])

        return self.object_type_to_ind[task.task_info["object_type"]]
