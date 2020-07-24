import typing
from typing import Any, Dict, Optional, List

import gym
import numpy as np

from rl_ai2thor.ai2thor_environment import AI2ThorEnvironment
from rl_ai2thor.object_nav.tasks import ObjectNavTask
from rl_base.sensor import Sensor, RGBSensor
from rl_base.task import Task


class RGBSensorThor(RGBSensor[AI2ThorEnvironment, Task[AI2ThorEnvironment]]):
    """Sensor for RGB images in AI2-THOR.

    Returns from a running AI2ThorEnvironment instance, the current RGB
    frame corresponding to the agent's egocentric view.
    """

    def frame_from_env(self, env: AI2ThorEnvironment) -> np.ndarray:
        return env.current_frame.copy()


class GoalObjectTypeThorSensor(Sensor):
    def __init__(self, config: Dict[str, Any], *args: Any, **kwargs: Any):
        super().__init__(config, *args, **kwargs)

        self.ordered_object_types: List[str] = list(self.config["object_types"])
        assert self.ordered_object_types == list(sorted(self.ordered_object_types)), (
            "object types" "input to goal object type " "sensor must be ordered"
        )

        if "target_to_detector_map" not in self.config:
            self.object_type_to_ind = {
                ot: i for i, ot in enumerate(self.ordered_object_types)
            }

            self.observation_space = gym.spaces.Discrete(len(self.ordered_object_types))
        else:
            assert (
                "detector_types" in self.config
            ), "Missing detector_types for map {}".format(
                self.config["target_to_detector_map"]
            )
            self.target_to_detector = self.config["target_to_detector_map"]
            self.detector_types = self.config["detector_types"]

            detector_index = {ot: i for i, ot in enumerate(self.detector_types)}
            self.object_type_to_ind = {
                ot: detector_index[self.target_to_detector[ot]]
                for ot in self.ordered_object_types
            }

            self.observation_space = gym.spaces.Discrete(len(self.detector_types))

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return "goal_object_type_ind"

    def _get_observation_space(self) -> gym.spaces.Discrete:
        return typing.cast(gym.spaces.Discrete, self.observation_space)

    def get_observation(
        self,
        env: AI2ThorEnvironment,
        task: Optional[ObjectNavTask],
        *args: Any,
        **kwargs: Any
    ) -> Any:

        # # Debug
        # print(task.task_info["object_type"], '->',
        #       self.detector_types[self.object_type_to_ind[task.task_info["object_type"]]])

        return self.object_type_to_ind[task.task_info["object_type"]]
