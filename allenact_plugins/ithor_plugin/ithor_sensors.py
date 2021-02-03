from typing import Any, Dict, Optional, List, Union, Sequence

import gym
import numpy as np

from allenact.base_abstractions.sensor import Sensor, RGBSensor
from allenact.base_abstractions.task import Task
from allenact.utils.misc_utils import prepare_locals_for_super
from allenact_plugins.ithor_plugin.ithor_environment import IThorEnvironment
from allenact_plugins.ithor_plugin.ithor_tasks import ObjectNaviThorGridTask
from allenact_plugins.robothor_plugin.robothor_environment import RoboThorEnvironment
from allenact_plugins.robothor_plugin.robothor_tasks import PointNavTask, ObjectNavTask


class RGBSensorThor(
    RGBSensor[
        Union[IThorEnvironment, RoboThorEnvironment],
        Union[Task[IThorEnvironment], Task[RoboThorEnvironment]],
    ]
):
    """Sensor for RGB images in THOR.

    Returns from a running IThorEnvironment instance, the current RGB
    frame corresponding to the agent's egocentric view.
    """

    def frame_from_env(
        self, env: IThorEnvironment, task: Task[IThorEnvironment]
    ) -> np.ndarray:  # type:ignore
        return env.current_frame.copy()


class GoalObjectTypeThorSensor(Sensor):
    def __init__(
        self,
        object_types: Sequence[str],
        target_to_detector_map: Optional[Dict[str, str]] = None,
        detector_types: Optional[Sequence[str]] = None,
        uuid: str = "goal_object_type_ind",
        **kwargs: Any
    ):
        self.ordered_object_types = list(object_types)
        assert self.ordered_object_types == sorted(
            self.ordered_object_types
        ), "object types input to goal object type sensor must be ordered"

        self.target_to_detector_map = target_to_detector_map

        if target_to_detector_map is None:
            self.object_type_to_ind = {
                ot: i for i, ot in enumerate(self.ordered_object_types)
            }
        else:
            assert (
                detector_types is not None
            ), "Missing detector_types for map {}".format(target_to_detector_map)
            self.target_to_detector = target_to_detector_map
            self.detector_types = detector_types

            detector_index = {ot: i for i, ot in enumerate(self.detector_types)}
            self.object_type_to_ind = {
                ot: detector_index[self.target_to_detector[ot]]
                for ot in self.ordered_object_types
            }

        observation_space = self._get_observation_space()

        super().__init__(**prepare_locals_for_super(locals()))

    def _get_observation_space(self):
        if self.target_to_detector_map is None:
            return gym.spaces.Discrete(len(self.ordered_object_types))
        else:
            return gym.spaces.Discrete(len(self.detector_types))

    def get_observation(
        self,
        env: IThorEnvironment,
        task: Optional[ObjectNaviThorGridTask],
        *args: Any,
        **kwargs: Any
    ) -> Any:
        return self.object_type_to_ind[task.task_info["object_type"]]


class TakeEndActionThorNavSensor(
    Sensor[
        Union[RoboThorEnvironment, IThorEnvironment],
        Union[ObjectNaviThorGridTask, ObjectNavTask, PointNavTask],
    ]
):
    def __init__(self, nactions: int, uuid: str, **kwargs: Any) -> None:
        self.nactions = nactions

        observation_space = self._get_observation_space()

        super().__init__(**prepare_locals_for_super(locals()))

    def _get_observation_space(self) -> gym.spaces.Discrete:
        """The observation space.

        Equals `gym.spaces.Discrete(2)` where a 0 indicates that the agent
        **should not** take the `End` action and a 1 indicates that the agent
        **should** take the end action.
        """
        return gym.spaces.Discrete(2)

    def get_observation(  # type:ignore
        self,
        env: IThorEnvironment,
        task: Union[ObjectNaviThorGridTask, ObjectNavTask, PointNavTask],
        *args,
        **kwargs
    ) -> np.ndarray:
        if isinstance(task, ObjectNaviThorGridTask):
            should_end = task.is_goal_object_visible()
        elif isinstance(task, ObjectNavTask):
            should_end = task._is_goal_in_range()
        elif isinstance(task, PointNavTask):
            should_end = task._is_goal_in_range()
        else:
            raise NotImplementedError

        if should_end is None:
            should_end = False
        return np.array([1 * should_end], dtype=np.int64)
