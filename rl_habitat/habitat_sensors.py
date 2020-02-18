from typing import Any, Dict, Optional

import PIL
import gym
import numpy as np
import typing
from PIL import Image
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms

from rl_habitat.habitat_environment import HabitatEnvironment
from rl_habitat.habitat_tasks import HabitatTask, PointNavTask
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


class RGBSensorHabitat(Sensor[HabitatEnvironment, PointNavTask]):
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

        self.norm_means = np.array([[[0.485, 0.456, 0.406]]], dtype=np.float32)
        self.norm_sds = np.array([[[0.229, 0.224, 0.225]]], dtype=np.float32)

        shape = None if self.height is None else (self.height, self.width, 3)
        if not self.should_normalize:
            low = 0.0
            high = 1.0
            self.observation_space = gym.spaces.Box(low=low, high=high, shape=shape)
        else:
            low = np.tile(-self.norm_means / self.norm_sds, shape[:-1] + (1,))
            high = np.tile((1 - self.norm_means) / self.norm_sds, shape[:-1] + (1,))
            self.observation_space = gym.spaces.Box(low=low, high=high)

        self.scaler = (
            None
            if self.width is None
            else ScaleBothSides(width=self.width, height=self.height)
        )

        self.to_pil = transforms.ToPILImage(mode='RGB')

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return "rgb"

    def _get_observation_space(self) -> gym.spaces.Box:
        return self.observation_space

    def get_observation(
            self,
            env: HabitatEnvironment,
            task: Optional[HabitatTask],
            *args: Any,
            **kwargs: Any
    ) -> Any:
        frame = env.current_frame
        rgb = frame["rgb"].copy()

        assert rgb.dtype in [np.uint8, np.float32]

        if rgb.dtype == np.uint8:
            rgb = rgb.astype(np.float32) / 255.0

        if self.should_normalize:
            rgb -= self.norm_means
            rgb /= self.norm_sds

        if self.scaler is not None and rgb.shape[:2] != (self.height, self.width):
            rgb = np.array(self.scaler(self.to_pil(rgb)), dtype=np.float32)

        return rgb


class RGBResNetSensorHabitat(Sensor[HabitatEnvironment, PointNavTask]):
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

        self.norm_means = np.array([[[0.485, 0.456, 0.406]]], dtype=np.float32)
        self.norm_sds = np.array([[[0.229, 0.224, 0.225]]], dtype=np.float32)

        shape = None if self.height is None else (2048, )
        if not self.should_normalize:
            low = 0.0
            high = 1.0
            self.observation_space = gym.spaces.Box(low=low, high=high, shape=shape)
        else:
            low = np.tile(-self.norm_means / self.norm_sds, shape[:-1] + (1,))
            high = np.tile((1 - self.norm_means) / self.norm_sds, shape[:-1] + (1,))
            self.observation_space = gym.spaces.Box(low=low, high=high)

        self.scaler = (
            None
            if self.width is None
            else ScaleBothSides(width=self.width, height=self.height)
        )

        self.to_pil = transforms.ToPILImage(mode='RGB')
        self.to_tensor = transforms.ToTensor()

        self.resnet = nn.Sequential(
            *list(models.resnet50(pretrained=True).children())[:-1] + [nn.Flatten()]
        ).eval()

        if torch.cuda.is_available():
            self.resnet = self.resnet.to("cuda:0")

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return "rgb"

    def _get_observation_space(self) -> gym.spaces.Box:
        return self.observation_space

    def get_observation(
            self,
            env: HabitatEnvironment,
            task: Optional[HabitatTask],
            *args: Any,
            **kwargs: Any
    ) -> Any:
        frame = env.current_frame
        rgb = frame["rgb"].copy()

        assert rgb.dtype in [np.uint8, np.float32]

        if rgb.dtype == np.uint8:
            rgb = rgb.astype(np.float32) / 255.0

        if self.should_normalize:
            rgb -= self.norm_means
            rgb /= self.norm_sds

        if self.scaler is not None and rgb.shape[:2] != (self.height, self.width):
            rgb = np.array(self.scaler(self.to_pil(rgb)), dtype=np.float32)

        rgb = self.to_tensor(rgb).unsqueeze(0)
        if torch.cuda.is_available():
            rgb = rgb.to("cuda:0")
        rgb = self.resnet(rgb).detach().cpu().numpy()

        return rgb


class DepthSensorHabitat(Sensor[HabitatEnvironment, PointNavTask]):
    def __init__(self, config: Dict[str, Any], *args: Any, **kwargs: Any):
        super().__init__(config, *args, **kwargs)

        def f(x, k, default):
            return x[k] if k in x else default

        self.height: Optional[int] = f(config, "height", None)
        self.width: Optional[int] = f(config, "width", None)
        self.should_normalize = f(config, "use_resnet_normalization", False)

        assert (self.width is None) == (self.height is None), (
            "In RGBSensorThor's config, "
            "either both height/width must be None or neither."
        )

        self.norm_means = np.array([0.5], dtype=np.float32)
        self.norm_sds = np.array([[0.25]], dtype=np.float32)

        shape = None if self.height is None else (self.height, self.width, 3)
        if not self.should_normalize:
            low = 0.0
            high = 1.0
            self.observation_space = gym.spaces.Box(low=low, high=high, shape=shape)
        else:
            low = np.tile(-self.norm_means / self.norm_sds, shape[:-1] + (1,))
            high = np.tile((1 - self.norm_means) / self.norm_sds, shape[:-1] + (1,))
            self.observation_space = gym.spaces.Box(low=low, high=high)

        self.scaler = (
            None
            if self.width is None
            else ScaleBothSides(width=self.width, height=self.height)
        )

        self.to_pil = transforms.ToPILImage()

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return "depth"

    def _get_observation_space(self) -> gym.spaces.Box:
        return self.observation_space

    def get_observation(
            self,
            env: HabitatEnvironment,
            task: Optional[HabitatTask],
            *args: Any,
            **kwargs: Any
    ) -> Any:
        frame = env.current_frame
        depth = frame["depth"].copy()

        assert depth.dtype in [np.uint8, np.float32]

        if depth.dtype == np.uint8:
            depth = depth.astype(np.float32) / 255.0

        if self.should_normalize:
            depth -= self.norm_means
            depth /= self.norm_sds

        if self.scaler is not None and depth.shape[:2] != (self.height, self.width):
            depth = np.array(self.scaler(self.to_pil(depth)), dtype=np.float32)

        depth = np.expand_dims(depth, 2)

        return depth


class DepthResNetSensorHabitat(Sensor[HabitatEnvironment, PointNavTask]):
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

        self.norm_means = np.array([0.5], dtype=np.float32)
        self.norm_sds = np.array([[0.25]], dtype=np.float32)

        shape = None if self.height is None else (2048, )
        if not self.should_normalize:
            low = 0.0
            high = 1.0
            self.observation_space = gym.spaces.Box(low=low, high=high, shape=shape)
        else:
            low = np.tile(-self.norm_means / self.norm_sds, shape[:-1] + (1,))
            high = np.tile((1 - self.norm_means) / self.norm_sds, shape[:-1] + (1,))
            self.observation_space = gym.spaces.Box(low=low, high=high)

        self.scaler = (
            None
            if self.width is None
            else ScaleBothSides(width=self.width, height=self.height)
        )

        self.to_pil = transforms.ToPILImage()
        self.to_tensor = transforms.ToTensor()

        self.resnet = nn.Sequential(
            *list(models.resnet50(pretrained=True).children())[:-1] + [nn.Flatten()]
        ).eval()

        if torch.cuda.is_available():
            self.resnet = self.resnet.to("cuda:0")

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return "depth"

    def _get_observation_space(self) -> gym.spaces.Box:
        return self.observation_space

    def get_observation(
            self,
            env: HabitatEnvironment,
            task: Optional[HabitatTask],
            *args: Any,
            **kwargs: Any
    ) -> Any:
        frame = env.current_frame
        depth = frame["depth"].copy()

        assert depth.dtype in [np.uint8, np.float32]

        if depth.dtype == np.uint8:
            rgb = depth.astype(np.float32) / 255.0

        if self.should_normalize:
            depth -= self.norm_means
            depth /= self.norm_sds

        if self.scaler is not None and depth.shape[:2] != (self.height, self.width):
            depth = np.array(self.scaler(self.to_pil(depth)), dtype=np.float32)

        depth = self.to_tensor(depth).squeeze()
        depth = torch.cat([3 * depth], dim=2).unsqueeze(0)
        if torch.cuda.is_available():
            depth = depth.to("cuda:0")
        depth = self.resnet(depth).detach().cpu().numpy()

        return depth


class TargetCoordinatesSensorHabitat(Sensor[HabitatEnvironment, PointNavTask]):
    def __init__(self, config: Dict[str, Any], *args: Any, **kwargs: Any):
        super().__init__(config, *args, **kwargs)

        self.observation_space = gym.spaces.Discrete(config["coordinate_dims"])

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return "target_coordinates_ind"

    def _get_observation_space(self) -> gym.spaces.Discrete:
        return typing.cast(gym.spaces.Discrete, self.observation_space)

    def get_observation(
        self,
        env: HabitatEnvironment,
        task: Optional[PointNavTask],
        *args: Any,
        **kwargs: Any
    ) -> Any:
        frame = env.current_frame
        goal = frame["pointgoal_with_gps_compass"]
        return goal
