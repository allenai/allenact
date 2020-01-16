# Original work Copyright (c) Facebook, Inc. and its affiliates.
# Modified work Copyright (c) Allen Institute for AI
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import OrderedDict
from typing import Generic, Dict, Any, List, Optional

import gym
from gym.spaces import Dict as SpaceDict

from rl_base.common import EnvType
from rl_base.task import Task


class Sensor(Generic[EnvType]):
    """Represents a sensor that provides data from the environment to agent.
    The user of this class needs to implement the get_observation method and
    the user is also required to set the below attributes:

    Attributes:
        config: configuration information for the sensor.
        uuid: universally unique id.
        observation_space: ``gym.Space`` object corresponding to observation of
            sensor.
    """

    config: Dict[str, Any]
    uuid: str
    observation_space: gym.Space

    def __init__(self, config: Dict[str, Any], *args: Any, **kwargs: Any) -> None:
        self.config = config
        self.uuid = self._get_uuid()
        self.observation_space = self._get_observation_space()

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        raise NotImplementedError()

    def _get_observation_space(self) -> gym.Space:
        raise NotImplementedError()

    def get_observation(
        self, env: EnvType, task: Optional[Task], *args: Any, **kwargs: Any
    ) -> Any:
        """
        Returns:
            current observation for Sensor.
        """
        raise NotImplementedError()


class SensorSuite(Generic[EnvType]):
    """Represents a set of sensors, with each sensor being identified through a
    unique id.

    Args:
        sensors: list containing sensors for the environment, uuid of each
            sensor must be unique.
    """

    sensors: Dict[str, Sensor[EnvType]]
    observation_spaces: SpaceDict

    def __init__(self, sensors: List[Sensor]) -> None:
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
        return self.sensors[uuid]

    def get_observations(
        self, env: EnvType, task: Optional[Task[EnvType]], *args: Any, **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Returns:
            collect data from all sensors and return it packaged inside
            a Dict.
        """
        return {
            uuid: sensor.get_observation(env=env, task=task, *args, **kwargs)
            for uuid, sensor in self.sensors.items()
        }
