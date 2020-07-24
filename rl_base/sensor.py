# Original work Copyright (c) Facebook, Inc. and its affiliates.
# Modified work Copyright (c) Allen Institute for AI
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import OrderedDict
from typing import Generic, Dict, Any, Optional, TYPE_CHECKING, TypeVar, Sequence, cast
from abc import abstractmethod

import gym
from gym.spaces import Dict as SpaceDict
import torch
from torchvision import transforms, models
from torch import nn

from rl_base.common import EnvType
from utils.tensor_utils import ScaleBothSides

if TYPE_CHECKING:
    from rl_base.task import SubTaskType
else:
    SubTaskType = TypeVar("SubTaskType", bound="Task")

import numpy as np


class Sensor(Generic[EnvType, SubTaskType]):
    """Represents a sensor that provides data from the environment to agent.
    The user of this class needs to implement the get_observation method and
    the user is also required to set the below attributes:

    # Attributes

    config : configuration information for the sensor.
    uuid : universally unique id.
    observation_space : ``gym.Space`` object corresponding to observation of
        sensor.
    """

    config: Dict[str, Any]
    uuid: str
    observation_space: gym.Space

    def __init__(self, config: Dict[str, Any], *args: Any, **kwargs: Any) -> None:
        self.config = config
        self.uuid = self._get_uuid()

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        """The unique ID of the sensor.

        # Parameters

        args : extra args.
        kwargs : extra kwargs.
        """
        raise NotImplementedError()

    def _get_observation_space(self) -> gym.Space:
        """The observation space of the sensor."""
        raise NotImplementedError()

    def get_observation(
        self, env: EnvType, task: Optional[SubTaskType], *args: Any, **kwargs: Any
    ) -> Any:
        """Returns observations from the environment (or task).

        # Parameters

        env : The environment the sensor is used upon.
        task : (Optionally) a Task from which the sensor should get data.

        # Returns

        Current observation for Sensor.
        """
        raise NotImplementedError()


class SensorSuite(Generic[EnvType]):
    """Represents a set of sensors, with each sensor being identified through a
    unique id.

    # Attributes

    sensors: list containing sensors for the environment, uuid of each
        sensor must be unique.
    """

    sensors: Dict[str, Sensor[EnvType, Any]]
    observation_spaces: SpaceDict

    def __init__(self, sensors: Sequence[Sensor]) -> None:
        """Initializer.

        # Parameters

        param sensors: the sensors that will be included in the suite.
        """
        self.sensors = OrderedDict()
        spaces: OrderedDict[str, gym.Space] = OrderedDict()
        for sensor in sensors:
            assert (
                sensor.uuid not in self.sensors
            ), "'{}' is duplicated sensor uuid".format(sensor.uuid)
            self.sensors[sensor.uuid] = sensor
            spaces[sensor.uuid] = sensor.observation_space
        self.observation_spaces = SpaceDict(spaces=spaces)

    def get(self, uuid: str) -> Sensor:
        """Return sensor with the given `uuid`.

        # Parameters

        uuid : The unique id of the sensor

        # Returns

        The sensor with unique id `uuid`.
        """
        return self.sensors[uuid]

    def get_observations(
        self, env: EnvType, task: Optional[SubTaskType], **kwargs: Any
    ) -> Dict[str, Any]:
        """Get all observations corresponding to the sensors in the suite.

        # Parameters

        env : The environment from which to get the observation.
        task : (Optionally) the task from which to get the observation.

        # Returns

        Data from all sensors packaged inside a Dict.
        """
        return {
            uuid: sensor.get_observation(env=env, task=task, **kwargs)  # type: ignore
            for uuid, sensor in self.sensors.items()
        }


class ExpertActionSensor(Sensor[EnvType, SubTaskType]):
    def __init__(self, config: Dict[str, Any], *args: Any, **kwargs: Any) -> None:
        super().__init__(config, *args, **kwargs)
        self.config = config
        self.uuid = self._get_uuid()
        self.observation_space = self._get_observation_space()
        self.expert_args: Dict[str, Any] = config.get("expert_args", {})

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return "expert_action"

    def _get_observation_space(self) -> gym.spaces.Tuple:
        """The observation space of the expert action sensor.

        Will equal `gym.spaces.Tuple(gym.spaces.Discrete(num actions in
        task), gym.spaces.Discrete(2))` where the first entry of the
        tuple is the expert action index and the second equals 0 if and
        only if the expert failed to generate a true expert action. The
        value `num actions in task` should be in `config["nactions"]`
        """
        return gym.spaces.Tuple(
            (gym.spaces.Discrete(self.config["nactions"]), gym.spaces.Discrete(2))
        )

    def get_observation(
        self, env: EnvType, task: SubTaskType, *args: Any, **kwargs: Any
    ) -> Any:
        # If the task is completed, we needn't (perhaps can't) find the expert
        # action from the (current) terminal state.
        if task.is_done():
            return np.array([-1, False], dtype=np.int64)
        action, expert_was_successful = task.query_expert(**self.expert_args)
        assert isinstance(action, int), (
            "In expert action sensor, `task.query_expert()` "
            "did not return an integer action."
        )
        return np.array([action, expert_was_successful], dtype=np.int64)


class ExpertPolicySensor(Sensor[EnvType, SubTaskType]):
    def __init__(self, config: Dict[str, Any], *args: Any, **kwargs: Any) -> None:
        super().__init__(config, *args, **kwargs)
        self.config = config
        self.uuid = self._get_uuid()
        self.observation_space = self._get_observation_space()
        self.expert_args: Dict[str, Any] = config.get("expert_args", {})

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return "expert_policy"

    def _get_observation_space(self) -> gym.spaces.Tuple:
        """The observation space of the expert action sensor.

        Will equal `gym.spaces.Tuple(gym.spaces.Box(num actions in
        task), gym.spaces.Discrete(2))` where the first entry of the
        tuple is the expert policy and the second equals 0 if and only
        if the expert failed to generate a true expert action. The value
        `num actions in task` should be in `config["nactions"]`
        """
        return gym.spaces.Tuple(
            (
                gym.spaces.Box(
                    low=np.float32(0.0),
                    high=np.float32(1.0),
                    shape=(self.config["nactions"],),
                ),
                gym.spaces.Discrete(2),
            )
        )

    def get_observation(
        self, env: EnvType, task: SubTaskType, *args: Any, **kwargs: Any
    ) -> Any:
        policy, expert_was_successful = task.query_expert(**self.expert_args)
        assert isinstance(policy, np.ndarray) and policy.shape == (
            self.config["nactions"],
        ), (
            "In expert action sensor, `task.query_expert()` "
            "did not return an numpy array."
        )
        return np.array(
            np.concatenate((policy, [expert_was_successful]), axis=0), dtype=np.float32
        )


class VisionSensor(Sensor[EnvType, SubTaskType]):
    def __init__(
        self, config: Dict[str, Any], scale_first=True, *args: Any, **kwargs: Any
    ):
        """Initializer.

        # Parameters

        config : The images will be normalized
            with means `config["mean"]` and standard deviations `config["stdev"]`. If both `config["height"]` and
            `config["width"]` are non-negative integers then
            the image returned from the environment will be rescaled to have
            `config["height"]` rows and `config["width"]` columns using bilinear sampling.
        args : Extra args. Currently unused.
        kwargs : Extra kwargs. Currently unused.
        """

        def f(x, k, default):
            return x[k] if k in x else default

        self._norm_means = f(config, "NORM_mean", None)
        self._norm_sds = f(config, "NORM_stdev", None)
        assert (self._norm_means is None) == (self._norm_sds is None), (
            "In VisionSensor's config, "
            "either both mean/stdev must be None or neither."
        )
        self._should_normalize = self._norm_means is not None
        self._height: Optional[int] = f(config, "height", None)
        self._width: Optional[int] = f(config, "width", None)
        self._uuid: str = f(config, "uuid", "vision")
        self._scale_first = scale_first

        assert (self._width is None) == (self._height is None), (
            "In VisionSensor's config, "
            "either both height/width must be None or neither."
        )

        channels = f(config, "OUTPUT_CHANNELS", None)
        shape = None if self._height is None else (self._height, self._width, channels)
        low = f(config, "INPUT_LOW", -np.inf)
        high = f(config, "INPUT_HIGH", np.inf)

        if not self._should_normalize or shape is None:
            self.observation_space = gym.spaces.Box(
                low=np.float32(low), high=np.float32(high), shape=shape
            )
        else:
            low = np.tile((low - self._norm_means) / self._norm_sds, shape[:-1] + (1,))
            high = np.tile(
                (high - self._norm_means) / self._norm_sds, shape[:-1] + (1,)
            )
            self.observation_space = gym.spaces.Box(
                low=np.float32(low), high=np.float32(high)
            )

        self.scaler = (
            None
            if self._width is None
            else ScaleBothSides(
                width=cast(int, self._width), height=cast(int, self._height)
            )
        )

        self.to_pil = transforms.ToPILImage()  # assumes mode="RGB" for 3 channels

        super().__init__(
            config, *args, **kwargs
        )  # call it last so that user can assign a uuid

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

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self._uuid

    def _get_observation_space(self) -> gym.spaces.Box:
        return cast(gym.spaces.Box, self.observation_space)

    @abstractmethod
    def frame_from_env(self, env: EnvType):
        return NotImplementedError

    def get_observation(
        self, env: EnvType, task: Optional[SubTaskType], *args: Any, **kwargs: Any
    ) -> Any:
        im = self.frame_from_env(env)

        if self._scale_first:
            if self.scaler is not None and im.shape[:2] != (self._height, self._width):
                im = np.array(self.scaler(self.to_pil(im)), dtype=np.uint8)  # hwc

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


class RGBSensor(VisionSensor[EnvType, SubTaskType]):
    def __init__(
        self, config: Dict[str, Any], scale_first=True, *args: Any, **kwargs: Any
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

        def f(x, k, default):
            return x[k] if k in x else default

        config["uuid"] = f(config, "uuid", "rgb")
        config["OUTPUT_CHANNELS"] = 3
        config["INPUT_LOW"] = 0.0
        config["INPUT_HIGH"] = 1.0
        if f(config, "use_resnet_normalization", False):
            config["NORM_mean"] = np.array([[[0.485, 0.456, 0.406]]], dtype=np.float32)
            config["NORM_stdev"] = np.array([[[0.229, 0.224, 0.225]]], dtype=np.float32)

        super().__init__(config, scale_first, *args, **kwargs)

    @abstractmethod
    def frame_from_env(self, env: EnvType):
        return NotImplementedError


class DepthSensor(VisionSensor[EnvType, SubTaskType]):
    def __init__(
        self, config: Dict[str, Any], scale_first=True, *args: Any, **kwargs: Any
    ):
        def f(x, k, default):
            return x[k] if k in x else default

        config["uuid"] = f(config, "uuid", "depth")
        config["OUTPUT_CHANNELS"] = 1
        config["INPUT_LOW"] = 0.0
        config["INPUT_HIGH"] = 5.0
        config["NORM_mean"] = np.array([[0.5]], dtype=np.float32)
        config["NORM_stdev"] = np.array([[0.25]], dtype=np.float32)

        super().__init__(config, scale_first, *args, **kwargs)

    @abstractmethod
    def frame_from_env(self, env: EnvType):
        return NotImplementedError

    def get_observation(  # type: ignore
        self, env: EnvType, task: Optional[SubTaskType], *args: Any, **kwargs: Any
    ) -> Any:
        depth = super().get_observation(env, task, *args, **kwargs)
        depth = np.expand_dims(depth, 2)

        return depth


class RGBResNetSensor(VisionSensor[EnvType, SubTaskType]):
    def __init__(
        self, config: Dict[str, Any], scale_first=True, *args: Any, **kwargs: Any
    ):
        def f(x, k, default):
            return x[k] if k in x else default

        config["uuid"] = f(config, "uuid", "rgbresnet")
        if f(config, "use_resnet_normalization", True):
            config["NORM_mean"] = np.array([[[0.485, 0.456, 0.406]]], dtype=np.float32)
            config["NORM_stdev"] = np.array([[[0.229, 0.224, 0.225]]], dtype=np.float32)

        super().__init__(config, scale_first, *args, **kwargs)

        self.to_tensor = transforms.ToTensor()

        self.resnet = nn.Sequential(
            *list(models.resnet50(pretrained=True).children())[:-1] + [nn.Flatten()]
        ).eval()

        if torch.cuda.is_available():
            self.resnet = self.resnet.to("cuda")

    @abstractmethod
    def frame_from_env(self, env: EnvType):
        return NotImplementedError

    def get_observation(  # type: ignore
        self, env: EnvType, task: Optional[SubTaskType], *args: Any, **kwargs: Any
    ) -> Any:
        rgb = super().get_observation(env, task, *args, **kwargs)

        rgb = self.to_tensor(rgb).unsqueeze(0)
        if torch.cuda.is_available():
            rgb = rgb.to("cuda")
        rgb = self.resnet(rgb).detach().cpu().numpy()

        return rgb
