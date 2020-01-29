# Original work Copyright (c) Facebook, Inc. and its affiliates.
# Modified work Copyright (c) Allen Institute for AI
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import OrderedDict
from typing import Generic, Dict, Any, List, Optional, TYPE_CHECKING, TypeVar

import gym
from gym.spaces import Dict as SpaceDict

from rl_base.common import EnvType

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

    def __init__(self, sensors: List[Sensor]) -> None:
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
        self, env: EnvType, task: Optional[SubTaskType], *args: Any, **kwargs: Any
    ) -> Dict[str, Any]:
        """Get all observations corresponding to the sensors in the suite.

        # Attributes

        env : The environment from which to get the observation.
        task : (Optionally) the task from which to get the observation.

        # Returns

        Data from all sensors packaged inside a Dict.
        """
        return {
            uuid: sensor.get_observation(env=env, task=task, *args, **kwargs)  # type: ignore
            for uuid, sensor in self.sensors.items()
        }


class ExpertActionSensor(Sensor[EnvType, SubTaskType]):
    def __init__(self, config: Dict[str, Any], *args: Any, **kwargs: Any) -> None:
        super().__init__(config, *args, **kwargs)
        self.config = config
        self.uuid = self._get_uuid()
        self.observation_space = self._get_observation_space()

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
        action, expert_was_successful = task.query_expert()
        assert isinstance(action, int), (
            "In expert action sensor, `task.query_expert()` "
            "did not return an integer action."
        )
        return np.array([action, expert_was_successful], dtype=np.int64)
