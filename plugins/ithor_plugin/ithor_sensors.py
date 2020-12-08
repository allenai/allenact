from typing import Any, Dict, Optional, List, Union

import gym
import numpy as np

from core.base_abstractions.sensor import Sensor, RGBSensor
from core.base_abstractions.task import Task
from plugins.ithor_plugin.ithor_environment import IThorEnvironment
from plugins.ithor_plugin.ithor_tasks import ObjectNaviThorGridTask
from plugins.robothor_plugin.robothor_environment import RoboThorEnvironment
from plugins.robothor_plugin.robothor_tasks import PointNavTask, ObjectNavTask
from utils.misc_utils import prepare_locals_for_super


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

    def frame_from_env(self, env: IThorEnvironment) -> np.ndarray:
        return env.current_frame.copy()


class GoalObjectTypeThorSensor(Sensor):
    def __init__(
        self,
        object_types: List[str],
        target_to_detector_map: Optional[Dict[str, str]] = None,
        detector_types: Optional[List[str]] = None,
        uuid: str = "goal_object_type_ind",
        **kwargs: Any
    ):
        self.ordered_object_types = list(object_types)
        assert self.ordered_object_types == sorted(
            self.ordered_object_types
        ), "object types input to goal object type sensor must be ordered"

        if target_to_detector_map is None:
            self.object_type_to_ind = {
                ot: i for i, ot in enumerate(self.ordered_object_types)
            }

            observation_space = gym.spaces.Discrete(len(self.ordered_object_types))
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

            observation_space = gym.spaces.Discrete(len(self.detector_types))

        super().__init__(**prepare_locals_for_super(locals()))

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

    def get_observation(
        self,
        env: IThorEnvironment,
        task: Union[ObjectNaviThorGridTask, ObjectNavTask, PointNavTask],
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
