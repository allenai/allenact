# Original work Copyright (c) Facebook, Inc. and its affiliates.
# Modified work Copyright (c) Allen Institute for AI
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from collections import OrderedDict
from typing import (
    Generic,
    Dict,
    Any,
    Optional,
    TYPE_CHECKING,
    TypeVar,
    Sequence,
    Union,
    Tuple,
    cast,
)

import gym
import numpy as np
import gym.spaces as gyms
from torch.distributions.utils import lazy_property

from allenact.base_abstractions.misc import EnvType
from allenact.utils import spaces_utils as su
from allenact.utils.misc_utils import prepare_locals_for_super
from allenact.utils.system import get_logger

if TYPE_CHECKING:
    from allenact.base_abstractions.task import SubTaskType
else:
    SubTaskType = TypeVar("SubTaskType", bound="Task")

SpaceDict = gyms.Dict


class Sensor(Generic[EnvType, SubTaskType]):
    """Represents a sensor that provides data from the environment to agent.
    The user of this class needs to implement the get_observation method and
    the user is also required to set the below attributes:

    # Attributes

    uuid : universally unique id.
    observation_space : ``gym.Space`` object corresponding to observation of
        sensor.
    """

    uuid: str
    observation_space: gym.Space

    def __init__(self, uuid: str, observation_space: gym.Space, **kwargs: Any) -> None:
        self.uuid = uuid
        self.observation_space = observation_space

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
    observation_spaces: gyms.Dict

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
    """A sensor that obtains the expert action for a given task (if
    available)."""

    action_label: str = "action"
    expert_success_label: str = "expert_success"
    no_groups_label: str = "dummy_expert_group"

    def __init__(
        self,
        action_space: Optional[Union[gym.Space, int]] = None,
        uuid: str = "expert_action",
        expert_args: Optional[Dict[str, Any]] = None,
        nactions: Optional[int] = None,
        use_dict_as_groups: bool = True,
        **kwargs: Any
    ) -> None:
        """Initialize an `ExpertActionSensor`.

        # Parameters
        action_space : The action space of the agent, this is necessary in order for this sensor
            to know what its output observation space is.
        uuid : A string specifying the unique ID of this sensor.
        expert_args : This sensor obtains an expert action from the task by calling the `query_expert`
            method of the task. `expert_args` are any keyword arguments that should be passed to the
            `query_expert` method when called.
        nactions : [DEPRECATED] The number of actions available to the agent, corresponds to an `action_space`
            of `gym.spaces.Discrete(nactions)`.
        use_dict_as_groups : Whether to use the top-level action_space of type `gym.spaces.Dict` as action groups.
        """
        if isinstance(action_space, int):
            action_space = gym.spaces.Discrete(action_space)
        elif action_space is None:
            assert (
                nactions is not None
            ), "One of `action_space` or `nactions` must be not `None`."
            get_logger().warning(
                "The `nactions` parameter to `ExpertActionSensor` is deprecated and will be removed, please use"
                " the `action_space` parameter instead."
            )
            action_space = gym.spaces.Discrete(nactions)

        self.action_space = action_space

        self.use_groups = (
            isinstance(action_space, gym.spaces.Dict) and use_dict_as_groups
        )

        self.group_spaces = (
            self.action_space
            if self.use_groups
            else OrderedDict([(self.no_groups_label, self.action_space,)])
        )

        self.expert_args: Dict[str, Any] = expert_args or {}

        assert (
            "expert_sensor_action_group_name" not in self.expert_args
        ), "`expert_sensor_action_group_name` is reserved for `ExpertActionSensor`"

        observation_space = self._get_observation_space()

        super().__init__(**prepare_locals_for_super(locals()))

    @classmethod
    def flagged_group_space(cls, group_space: gym.spaces.Space) -> gym.spaces.Dict:
        return gym.spaces.Dict(
            [
                (cls.action_label, group_space),
                (cls.expert_success_label, gym.spaces.Discrete(2)),
            ]
        )

    @classmethod
    def flagged_space(
        cls, action_space: gym.spaces.Space, use_dict_as_groups: bool = True
    ) -> gym.spaces.Dict:
        use_groups = isinstance(action_space, gym.spaces.Dict) and use_dict_as_groups

        if not use_groups:
            return cls.flagged_group_space(action_space)
        else:
            return gym.spaces.Dict(
                [
                    (group_space, cls.flagged_group_space(action_space[group_space]),)
                    for group_space in cast(gym.spaces.Dict, action_space)
                ]
            )

    def _get_observation_space(self) -> Union[gym.spaces.Dict, gym.spaces.Tuple]:
        """The observation space of the expert action sensor.

        Will equal `gym.spaces.Tuple(gym.spaces.Discrete(num actions in
        task), gym.spaces.Discrete(2))` where the first entry of the
        tuple is the expert action index and the second equals 0 if and
        only if the expert failed to generate a true expert action. The
        value `num actions in task` should be in `config["nactions"]`
        """
        return self.flagged_space(self.action_space, use_dict_as_groups=self.use_groups)

    @lazy_property
    def _zeroed_observation(self) -> Union[OrderedDict, Tuple]:
        # AllenAct-style flattened space (to easily generate an all-zeroes action as an array)
        flat_space = su.flatten_space(self.observation_space)
        # torch point to correctly unflatten `Discrete` for zeroed output
        flat_zeroed = su.torch_point(flat_space, np.zeros_like(flat_space.sample()))
        # unflatten zeroed output and convert to numpy
        return su.numpy_point(
            self.observation_space, su.unflatten(self.observation_space, flat_zeroed)
        )

    def flatten_output(self, unflattened):
        return (
            su.flatten(
                self.observation_space,
                su.torch_point(self.observation_space, unflattened),
            )
            .cpu()
            .numpy()
        )

    def get_observation(
        self, env: EnvType, task: SubTaskType, *args: Any, **kwargs: Any
    ) -> Union[OrderedDict, Tuple]:
        # If the task is completed, we needn't (perhaps can't) find the expert
        # action from the (current) terminal state.
        if task.is_done():
            return self.flatten_output(self._zeroed_observation)

        actions = OrderedDict()
        for group_name in self.group_spaces:
            action, expert_was_successful = task.query_expert(
                **self.expert_args, expert_sensor_action_group_name=group_name
            )

            if isinstance(action, int):
                assert isinstance(self.group_spaces[group_name], gym.spaces.Discrete)
                unflattened_action = action
            else:
                # Assume we receive a gym-flattened numpy action
                unflattened_action = gyms.unflatten(
                    self.group_spaces[group_name], action
                )
                # TODO why not enforcing unflattened actions from the task?

            actions[group_name] = OrderedDict(
                [
                    (self.action_label, unflattened_action),
                    (self.expert_success_label, expert_was_successful),
                ]
            )

        return self.flatten_output(
            actions if self.use_groups else actions[self.no_groups_label]
        )


class ExpertPolicySensor(Sensor[EnvType, SubTaskType]):
    def __init__(
        self,
        nactions: int,
        uuid: str = "expert_policy",
        expert_args: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> None:
        self.nactions = nactions
        self.expert_args: Dict[str, Any] = expert_args or {}

        super().__init__(**prepare_locals_for_super(locals()))

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
                    low=np.float32(0.0), high=np.float32(1.0), shape=(self.nactions,),
                ),
                gym.spaces.Discrete(2),
            )
        )

    def get_observation(
        self, env: EnvType, task: SubTaskType, *args: Any, **kwargs: Any
    ) -> Any:
        policy, expert_was_successful = task.query_expert(**self.expert_args)
        assert isinstance(policy, np.ndarray) and policy.shape == (self.nactions,), (
            "In expert action sensor, `task.query_expert()` "
            "did not return a valid numpy array."
        )
        return np.array(
            np.concatenate((policy, [expert_was_successful]), axis=-1), dtype=np.float32
        )
