from typing import Any, Dict, Optional, List

import gym
import numpy as np

from plugins.ithor_plugin.ithor_environment import IThorEnvironment
from plugins.ithor_plugin.ithor_tasks import ObjectNavTask
from core.base_abstractions.sensor import Sensor, RGBSensor
from core.base_abstractions.task import Task
from utils.misc_utils import prepare_locals_for_super


class RGBSensorThor(RGBSensor[IThorEnvironment, Task[IThorEnvironment]]):
    """Sensor for RGB images in AI2-THOR.

    Returns from a running IThorEnvironment instance, the current RGB
    frame corresponding to the agent's egocentric view.
    """

    def frame_from_env(self, env: IThorEnvironment) -> np.ndarray:
        return env.current_frame.copy()


class GoalObjectTypeThorSensor(Sensor):
    def __init__(
        self,
        object_types: List[str],
        target_to_detector_map: Optional[Dict[str, int]] = None,
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
        task: Optional[ObjectNavTask],
        *args: Any,
        **kwargs: Any
    ) -> Any:
        return self.object_type_to_ind[task.task_info["object_type"]]
