from abc import abstractmethod, ABC
from typing import Optional, Tuple, Any, cast

import PIL
import gym
import numpy as np
from torchvision import transforms

from allenact.base_abstractions.misc import EnvType
from allenact.base_abstractions.sensor import Sensor
from allenact.base_abstractions.task import SubTaskType
from allenact.utils.misc_utils import prepare_locals_for_super
from allenact.utils.tensor_utils import ScaleBothSides


class VisionSensor(Sensor[EnvType, SubTaskType]):
    def __init__(
        self,
        mean: Optional[np.ndarray] = None,
        stdev: Optional[np.ndarray] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        uuid: str = "vision",
        output_shape: Optional[Tuple[int, ...]] = None,
        output_channels: Optional[int] = None,
        unnormalized_infimum: float = -np.inf,
        unnormalized_supremum: float = np.inf,
        scale_first: bool = True,
        **kwargs: Any
    ):
        """Initializer.

        # Parameters

        config : The images will be normalized
            with means `config["mean"]` and standard deviations `config["stdev"]`. If both `config["height"]` and
            `config["width"]` are non-negative integers then
            the image returned from the environment will be rescaled to have
            `config["height"]` rows and `config["width"]` columns using bilinear sampling. The universally unique
            identifier will be set as `config["uuid"]`.
        args : Extra args. Currently unused.
        kwargs : Extra kwargs. Currently unused.
        """

        self._norm_means = mean
        self._norm_sds = stdev
        assert (self._norm_means is None) == (self._norm_sds is None), (
            "In VisionSensor's config, "
            "either both mean/stdev must be None or neither."
        )
        self._should_normalize = self._norm_means is not None

        self._height = height
        self._width = width
        assert (self._width is None) == (self._height is None), (
            "In VisionSensor's config, "
            "either both height/width must be None or neither."
        )

        self._scale_first = scale_first

        self.scaler: Optional[ScaleBothSides] = None
        if self._width is not None:
            self.scaler = ScaleBothSides(
                width=cast(int, self._width), height=cast(int, self._height)
            )

        self.to_pil = transforms.ToPILImage()  # assumes mode="RGB" for 3 channels

        self._observation_space = self._make_observation_space(
            output_shape=output_shape,
            output_channels=output_channels,
            unnormalized_infimum=unnormalized_infimum,
            unnormalized_supremum=unnormalized_supremum,
        )

        assert int(PIL.__version__.split(".")[0]) != 7, (
            "We found that Pillow version >=7.* has broken scaling,"
            " please downgrade to version 6.2.1 or upgrade to >=8.0.0"
        )

        observation_space = self._get_observation_space()

        super().__init__(**prepare_locals_for_super(locals()))

    def _make_observation_space(
        self,
        output_shape: Optional[Tuple[int, ...]],
        output_channels: Optional[int],
        unnormalized_infimum: float,
        unnormalized_supremum: float,
    ) -> gym.spaces.Box:
        assert output_shape is None or output_channels is None, (
            "In VisionSensor's config, "
            "only one of output_shape and output_channels can be not None."
        )

        shape: Optional[Tuple[int, ...]] = None
        if output_shape is not None:
            shape = output_shape
        elif self._height is not None and output_channels is not None:
            shape = (
                cast(int, self._height),
                cast(int, self._width),
                cast(int, output_channels),
            )

        if not self._should_normalize or shape is None or len(shape) == 1:
            return gym.spaces.Box(
                low=np.float32(unnormalized_infimum),
                high=np.float32(unnormalized_supremum),
                shape=shape,
            )
        else:
            out_shape = shape[:-1] + (1,)
            low = np.tile(
                (unnormalized_infimum - cast(np.ndarray, self._norm_means))
                / cast(np.ndarray, self._norm_sds),
                out_shape,
            )
            high = np.tile(
                (unnormalized_supremum - cast(np.ndarray, self._norm_means))
                / cast(np.ndarray, self._norm_sds),
                out_shape,
            )
            return gym.spaces.Box(low=np.float32(low), high=np.float32(high))

    def _get_observation_space(self):
        return self._observation_space

    @property
    def height(self) -> Optional[int]:
        """Height that input image will be rescale to have.

        # Returns

        The height as a non-negative integer or `None` if no rescaling is done.
        """
        return self._height

    @property
    def width(self) -> Optional[int]:
        """Width that input image will be rescale to have.

        # Returns

        The width as a non-negative integer or `None` if no rescaling is done.
        """
        return self._width

    @abstractmethod
    def frame_from_env(self, env: EnvType, task: Optional[SubTaskType]) -> np.ndarray:
        raise NotImplementedError

    def get_observation(
        self, env: EnvType, task: Optional[SubTaskType], *args: Any, **kwargs: Any
    ) -> Any:
        im = self.frame_from_env(env=env, task=task)
        assert (
            im.dtype == np.float32 and (len(im.shape) == 2 or im.shape[-1] == 1)
        ) or (im.shape[-1] == 3 and im.dtype == np.uint8), (
            "Input frame must either have 3 channels and be of"
            " type np.uint8 or have one channel and be of type np.float32"
        )

        if self._scale_first:
            if self.scaler is not None and im.shape[:2] != (self._height, self._width):
                im = np.array(self.scaler(self.to_pil(im)), dtype=im.dtype)  # hwc

        assert im.dtype in [np.uint8, np.float32]

        if im.dtype == np.uint8:
            im = im.astype(np.float32) / 255.0

        if self._should_normalize:
            im -= self._norm_means
            im /= self._norm_sds

        if not self._scale_first:
            if self.scaler is not None and im.shape[:2] != (self._height, self._width):
                im = np.array(self.scaler(self.to_pil(im)), dtype=np.float32)  # hwc

        return im


class RGBSensor(VisionSensor[EnvType, SubTaskType], ABC):
    def __init__(
        self,
        use_resnet_normalization: bool = False,
        mean: Optional[np.ndarray] = np.array(
            [[[0.485, 0.456, 0.406]]], dtype=np.float32
        ),
        stdev: Optional[np.ndarray] = np.array(
            [[[0.229, 0.224, 0.225]]], dtype=np.float32
        ),
        height: Optional[int] = None,
        width: Optional[int] = None,
        uuid: str = "rgb",
        output_shape: Optional[Tuple[int, ...]] = None,
        output_channels: int = 3,
        unnormalized_infimum: float = 0.0,
        unnormalized_supremum: float = 1.0,
        scale_first: bool = True,
        **kwargs: Any
    ):
        """Initializer.

        # Parameters

        config : If `config["use_resnet_normalization"]` is `True` then the RGB images will be normalized
            with means `[0.485, 0.456, 0.406]` and standard deviations `[0.229, 0.224, 0.225]` (i.e. using the standard
            resnet normalization). If both `config["height"]` and `config["width"]` are non-negative integers then
            the RGB image returned from the environment will be rescaled to have shape
            (config["height"], config["width"], 3) using bilinear sampling.
        args : Extra args. Currently unused.
        kwargs : Extra kwargs. Currently unused.
        """

        if not use_resnet_normalization:
            mean, stdev = None, None

        super().__init__(**prepare_locals_for_super(locals()))


class DepthSensor(VisionSensor[EnvType, SubTaskType], ABC):
    def __init__(
        self,
        use_normalization: bool = False,
        mean: Optional[np.ndarray] = np.array([[0.5]], dtype=np.float32),
        stdev: Optional[np.ndarray] = np.array([[0.25]], dtype=np.float32),
        height: Optional[int] = None,
        width: Optional[int] = None,
        uuid: str = "depth",
        output_shape: Optional[Tuple[int, ...]] = None,
        output_channels: int = 1,
        unnormalized_infimum: float = 0.0,
        unnormalized_supremum: float = 5.0,
        scale_first: bool = True,
        **kwargs: Any
    ):
        """Initializer.

        # Parameters

        config : If `config["use_normalization"]` is `True` then the depth images will be normalized
            with mean 0.5 and standard deviation 0.25. If both `config["height"]` and `config["width"]` are
            non-negative integers then the depth image returned from the environment will be rescaled to have shape
            (config["height"], config["width"]) using bilinear sampling.
        args : Extra args. Currently unused.
        kwargs : Extra kwargs. Currently unused.
        """

        if not use_normalization:
            mean, stdev = None, None

        super().__init__(**prepare_locals_for_super(locals()))

    def get_observation(  # type: ignore
        self, env: EnvType, task: Optional[SubTaskType], *args: Any, **kwargs: Any
    ) -> Any:
        depth = super().get_observation(env, task, *args, **kwargs)
        depth = np.expand_dims(depth, 2)

        return depth
