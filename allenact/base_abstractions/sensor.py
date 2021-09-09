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
import abc

import gym
import gym.spaces as gyms
import numpy as np
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


class AbstractExpertSensor(Sensor[EnvType, SubTaskType], abc.ABC):
    """Base class for sensors that obtain the expert action for a given task
    (if available)."""

    ACTION_POLICY_LABEL: str = "action_or_policy"
    EXPERT_SUCCESS_LABEL: str = "expert_success"
    _NO_GROUPS_LABEL: str = "__dummy_expert_group__"

    def __init__(
        self,
        action_space: Optional[Union[gym.Space, int]] = None,
        uuid: str = "expert_sensor_type_uuid",
        expert_args: Optional[Dict[str, Any]] = None,
        nactions: Optional[int] = None,
        use_dict_as_groups: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialize an `ExpertSensor`.

        # Parameters
        action_space : The action space of the agent. This is necessary in order for this sensor
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
                "The `nactions` parameter to `AbstractExpertSensor` is deprecated and will be removed, please use"
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
            else OrderedDict([(self._NO_GROUPS_LABEL, self.action_space,)])
        )

        self.expert_args: Dict[str, Any] = expert_args or {}

        assert (
            "expert_sensor_group_name" not in self.expert_args
        ), "`expert_sensor_group_name` is reserved for `AbstractExpertSensor`"

        observation_space = self._get_observation_space()

        super().__init__(**prepare_locals_for_super(locals()))

    @classmethod
    @abc.abstractmethod
    def flagged_group_space(cls, group_space: gym.spaces.Space) -> gym.spaces.Dict:
        """gym space resulting from wrapping the given action space (or a
        derived space, as in `AbstractExpertPolicySensor`) together with a
        binary action space corresponding to an expert success flag, in a Dict
        space.

        # Parameters
        group_space : The source action space to be (optionally used to derive a policy space,) flagged and wrapped
        """
        raise NotImplementedError

    @classmethod
    def flagged_space(
        cls, action_space: gym.spaces.Space, use_dict_as_groups: bool = True
    ) -> gym.spaces.Dict:
        """gym space resulting from wrapping the given action space (or every
        highest-level entry in a Dict action space), together with binary
        action space corresponding to an expert success flag, in a Dict space.

        # Parameters
        action_space : The agent's action space (to be flagged and wrapped)
        use_dict_as_groups : Flag enabling every highest-level entry in a Dict action space to be independently flagged.
        """
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

    def _get_observation_space(self) -> gym.spaces.Dict:
        """The observation space of the expert sensor.

        For the most basic discrete agent's ExpertActionSensor, it will
        equal `gym.spaces.Dict([ (self.ACTION_POLICY_LABEL,
        self.action_space), (self.EXPERT_SUCCESS_LABEL,
        gym.spaces.Discrete(2))])`, where the first entry hosts the
        expert action index and the second equals 0 if and only if the
        expert failed to generate a true expert action.
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

    @abc.abstractmethod
    def query_expert(
        self, task: SubTaskType, expert_sensor_group_name: Optional[str],
    ) -> Tuple[Any, bool]:
        """Query the expert for the given task (and optional group name).

        # Returns

         A tuple (x, y) where x is the expert action or policy and y is False \
            if the expert could not determine the optimal action (otherwise True). Here y \
            is used for masking. Even when y is False, x should still lie in the space of \
            possible values (e.g. if x is the expert policy then x should be the correct length, \
            sum to 1, and have non-negative entries).
        """
        raise NotImplementedError

    def get_observation(
        self, env: EnvType, task: SubTaskType, *args: Any, **kwargs: Any
    ) -> Union[OrderedDict, Tuple]:
        # If the task is completed, we needn't (perhaps can't) find the expert
        # action from the (current) terminal state.
        if task.is_done():
            return self.flatten_output(self._zeroed_observation)

        actions_or_policies = OrderedDict()
        for group_name in self.group_spaces:
            action_or_policy, expert_was_successful = self.query_expert(
                task=task, expert_sensor_group_name=group_name
            )

            actions_or_policies[group_name] = OrderedDict(
                [
                    (self.ACTION_POLICY_LABEL, action_or_policy),
                    (self.EXPERT_SUCCESS_LABEL, expert_was_successful),
                ]
            )

        return self.flatten_output(
            actions_or_policies
            if self.use_groups
            else actions_or_policies[self._NO_GROUPS_LABEL]
        )


class AbstractExpertActionSensor(AbstractExpertSensor, abc.ABC):
    def __init__(
        self,
        action_space: Optional[Union[gym.Space, int]] = None,
        uuid: str = "expert_action",
        expert_args: Optional[Dict[str, Any]] = None,
        nactions: Optional[int] = None,
        use_dict_as_groups: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**prepare_locals_for_super(locals()))

    @classmethod
    def flagged_group_space(cls, group_space: gym.spaces.Space) -> gym.spaces.Dict:
        """gym space resulting from wrapping the given action space, together
        with a binary action space corresponding to an expert success flag, in
        a Dict space.

        # Parameters
        group_space : The action space to be flagged and wrapped
        """
        return gym.spaces.Dict(
            [
                (cls.ACTION_POLICY_LABEL, group_space),
                (cls.EXPERT_SUCCESS_LABEL, gym.spaces.Discrete(2)),
            ]
        )


class ExpertActionSensor(AbstractExpertActionSensor):
    """(Deprecated) A sensor that obtains the expert action from a given task
    (if available)."""

    def query_expert(
        self, task: SubTaskType, expert_sensor_group_name: Optional[str]
    ) -> Tuple[Any, bool]:
        return task.query_expert(
            **self.expert_args, expert_sensor_group_name=expert_sensor_group_name
        )


class AbstractExpertPolicySensor(AbstractExpertSensor, abc.ABC):
    def __init__(
        self,
        action_space: Optional[Union[gym.Space, int]] = None,
        uuid: str = "expert_policy",
        expert_args: Optional[Dict[str, Any]] = None,
        nactions: Optional[int] = None,
        use_dict_as_groups: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**prepare_locals_for_super(locals()))

    @classmethod
    def flagged_group_space(cls, group_space: gym.spaces.Space) -> gym.spaces.Dict:
        """gym space resulting from wrapping the policy space corresponding to
        `allenact.utils.spaces_utils.policy_space(group_space)` together with a
        binary action space corresponding to an expert success flag, in a Dict
        space.

        # Parameters
        group_space : The source action space to be used to derive a policy space, flagged and wrapped
        """
        return gym.spaces.Dict(
            [
                (cls.ACTION_POLICY_LABEL, su.policy_space(group_space)),
                (cls.EXPERT_SUCCESS_LABEL, gym.spaces.Discrete(2)),
            ]
        )


class ExpertPolicySensor(AbstractExpertPolicySensor):
    """(Deprecated) A sensor that obtains the expert policy from a given task
    (if available)."""

    def query_expert(
        self, task: SubTaskType, expert_sensor_group_name: Optional[str]
    ) -> Tuple[Any, bool]:
        return task.query_expert(
            **self.expert_args, expert_sensor_group_name=expert_sensor_group_name
        )


class VisionSensor:
    def __init__(self, *args: Any, **kwargs: Any):
        raise ImportError(
            "`allenact.base_abstractions.sensor.VisionSensor` has moved!\n"
            "Please import allenact.embodiedai.sensors.vision_sensors.VisionSensor instead."
        )


class RGBSensor:
    def __init__(self, *args: Any, **kwargs: Any):
        raise ImportError(
            "`allenact.base_abstractions.sensor.RGBSensor` has moved!\n"
            "Please import allenact.embodiedai.sensors.vision_sensors.RGBSensor instead."
        )


class DepthSensor:
    def __init__(self, *args: Any, **kwargs: Any):
        raise ImportError(
            "`allenact.base_abstractions.sensor.DepthSensor` has moved!\n"
            "Please import allenact.embodiedai.sensors.vision_sensors.DepthSensor instead."
        )
