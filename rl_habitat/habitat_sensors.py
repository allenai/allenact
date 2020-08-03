from typing import Any, Optional, Tuple

import gym
import numpy as np
from pyquaternion import Quaternion

from rl_base.sensor import (
    Sensor,
    RGBSensor,
    RGBResNetSensor,
    DepthSensor,
    DepthResNetSensor,
)
from rl_base.task import Task
from rl_habitat.habitat_environment import HabitatEnvironment
from rl_habitat.habitat_tasks import PointNavTask  # type: ignore
from utils.misc_utils import prepare_locals_for_super


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


class RGBSensorHabitat(Sensor[HabitatEnvironment, HabitatTask]):
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


class RGBResNetSensorHabitat(Sensor[HabitatEnvironment, HabitatTask]):
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


class DepthSensorHabitat(Sensor[HabitatEnvironment, HabitatTask]):
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


class DepthResNetSensorHabitat(Sensor[HabitatEnvironment, HabitatTask]):
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

    def frame_from_env(self, env: HabitatEnvironment) -> np.ndarray:
        return env.current_frame["rgb"].copy()


class RGBResNetSensorHabitat(
    RGBResNetSensor[HabitatEnvironment, Task[HabitatEnvironment]]
):
    # For backwards compatibility
    def __init__(
        self,
        use_resnet_normalization: bool = True,
        mean: Optional[np.ndarray] = np.array(
            [[[0.485, 0.456, 0.406]]], dtype=np.float32
        ),
        stdev: Optional[np.ndarray] = np.array(
            [[[0.229, 0.224, 0.225]]], dtype=np.float32
        ),
        height: Optional[int] = None,
        width: Optional[int] = None,
        uuid: str = "rgb",
        output_shape: Optional[Tuple[int, ...]] = (2048,),
        output_channels: Optional[int] = None,
        unnormalized_infimum: float = -np.inf,
        unnormalized_supremum: float = np.inf,
        scale_first: bool = False,
        **kwargs: Any
    ):
        super().__init__(**prepare_locals_for_super(locals()))

    def frame_from_env(self, env: HabitatEnvironment) -> np.ndarray:
        return env.current_frame["rgb"].copy()


class DepthSensorHabitat(DepthSensor[HabitatEnvironment, Task[HabitatEnvironment]]):
    # For backwards compatibility
    def __init__(
        self,
        use_resnet_normalization: Optional[bool] = None,
        use_normalization: Optional[bool] = None,
        mean: Optional[np.ndarray] = np.array([[0.5]], dtype=np.float32),
        stdev: Optional[np.ndarray] = np.array([[0.25]], dtype=np.float32),
        height: Optional[int] = None,
        width: Optional[int] = None,
        uuid: str = "depth",
        output_shape: Optional[Tuple[int, ...]] = None,
        output_channels: int = 1,
        unnormalized_infimum: float = 0.0,
        unnormalized_supremum: float = 5.0,
        scale_first: bool = False,
        **kwargs: Any
    ):
        # Give priority to use_normalization, but use_resnet_normalization for backward compat. if not set
        if use_resnet_normalization is not None and use_normalization is None:
            use_normalization = use_resnet_normalization
        elif use_normalization is None:
            use_normalization = False

        super().__init__(**prepare_locals_for_super(locals()))

    def frame_from_env(self, env: HabitatEnvironment) -> np.ndarray:
        return env.current_frame["depth"].copy()


class DepthResNetSensorHabitat(
    DepthResNetSensor[HabitatEnvironment, Task[HabitatEnvironment]]
):
    # For backwards compatibility
    def __init__(
        self,
        use_resnet_normalization: Optional[bool] = None,
        use_normalization: Optional[bool] = None,
        mean: Optional[np.ndarray] = np.array([[0.5]], dtype=np.float32),
        stdev: Optional[np.ndarray] = np.array([[0.25]], dtype=np.float32),
        height: Optional[int] = None,
        width: Optional[int] = None,
        uuid: str = "depth",
        output_shape: Optional[Tuple[int, ...]] = (2048,),
        output_channels: Optional[int] = None,
        unnormalized_infimum: float = -np.inf,
        unnormalized_supremum: float = np.inf,
        scale_first: bool = False,
        **kwargs: Any
    ):
        # Give priority to use_normalization, but use_resnet_normalization for backward compat. if not set
        if use_resnet_normalization is not None and use_normalization is None:
            use_normalization = use_resnet_normalization
        elif use_normalization is None:
            use_normalization = False

        super().__init__(**prepare_locals_for_super(locals()))

    def frame_from_env(self, env: HabitatEnvironment) -> np.ndarray:
        return env.current_frame["depth"].copy()


class TargetCoordinatesSensorHabitat(Sensor[HabitatEnvironment, HabitatTask]):
    def __init__(self, config: Dict[str, Any], *args: Any, **kwargs: Any):
        super().__init__(config, *args, **kwargs)

        self.observation_space = gym.spaces.Box(-3.15, 1000, shape=(config["coordinate_dims"],))

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return "target_coordinates_ind"

        super().__init__(**prepare_locals_for_super(locals()))

    def get_observation(
        self,
        env: HabitatEnvironment,
        task: Optional[HabitatTask],
        *args: Any,
        **kwargs: Any
    ) -> Any:
        frame = env.current_frame
        goal = frame["pointgoal_with_gps_compass"]
        return goal

class TargetObjectSensorHabitat(Sensor[HabitatEnvironment, HabitatTask]):
    def __init__(self, config: Dict[str, Any], *args: Any, **kwargs: Any):
        super().__init__(config, *args, **kwargs)

        self.observation_space = gym.spaces.Discrete(38)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return "target_object_id"

        super().__init__(**prepare_locals_for_super(locals()))

    def get_observation(
        self,
        env: HabitatEnvironment,
        task: Optional[HabitatTask],
        *args: Any,
        **kwargs: Any
    ) -> Any:
        frame = env.current_frame
        goal = frame["objectgoal"][0]
        return goal

class AgentCoordinatesSensorHabitat(Sensor[HabitatEnvironment, HabitatTask]):
    def __init__(self, config: Dict[str, Any], *args: Any, **kwargs: Any):
        super().__init__(config, *args, **kwargs)

        self.observation_space = gym.spaces.Box(-1000, 1000, shape=(4,))

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return "agent_position_and_rotation"


        super().__init__(**prepare_locals_for_super(locals()))

    def get_observation(
        self,
        env: HabitatEnvironment,
        task: Optional[HabitatTask],
        *args: Any,
        **kwargs: Any
    ) -> Any:
        position = env.env.sim.get_agent_state().position
        quaternion = Quaternion(env.env.sim.get_agent_state().rotation.components)
        return np.array([position[0], position[1], position[2], quaternion.radians])
