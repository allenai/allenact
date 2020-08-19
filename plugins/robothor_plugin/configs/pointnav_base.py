from math import ceil
from typing import Dict, Any, List, Optional
import abc

import gym
import numpy as np

from core.base_abstractions.experiment_config import ExperimentConfig
from plugins.robothor_plugin.robothor_sensors import (
    GPSCompassSensorRoboThor,
    DepthSensorRoboThor,
)
from plugins.robothor_plugin.robothor_task_samplers import PointNavTaskSampler
from plugins.robothor_plugin.robothor_tasks import PointNavTask


class PointNavBaseConfig(ExperimentConfig, abc.ABC):
    """A Point Navigation base configuration."""

    CAMERA_WIDTH = 400
    CAMERA_HEIGHT = 300
    SCREEN_SIZE = 224

    VISION_UUID = "depth"
    TARGET_UUID = "target_coordinates_ind"

    MAX_STEPS = 500

    SENSORS = [
        DepthSensorRoboThor(
            use_normalization=True,
            height=SCREEN_SIZE,
            width=SCREEN_SIZE,
            uuid=VISION_UUID,
        ),
        GPSCompassSensorRoboThor(uuid=TARGET_UUID),
    ]

    ENV_ARGS = dict(
        width=CAMERA_WIDTH,
        height=CAMERA_HEIGHT,
        continuousMode=False,
        applyActionNoise=False,
        # agentType="stochastic",
        rotateStepDegrees=90.0,
        visibilityDistance=0.5,
        gridSize=0.25,
        snapToGrid=True,
        agentMode="bot",
        # include_private_scenes=True,
        renderDepthImage=True,
    )

    @classmethod
    def make_sampler_fn(cls, **kwargs):
        return PointNavTaskSampler(**kwargs)

    @staticmethod
    def _partition_inds(n: int, num_parts: int):
        return np.round(np.linspace(0, n, num_parts + 1, endpoint=True)).astype(
            np.int32
        )

    def _get_sampler_args_for_scene_split(
        self,
        scenes: List[str],
        process_ind: int,
        total_processes: int,
        seeds: Optional[List[int]] = None,
        deterministic_cudnn: bool = False,
    ) -> Dict[str, Any]:
        if total_processes > len(scenes):
            if total_processes % len(scenes) != 0:
                print(
                    "Warning: oversampling some of the scenes to feed all processes."
                    " You can avoid this by setting a number of workers divisible by the number of scenes"
                )
            scenes = scenes * int(ceil(total_processes / len(scenes)))
            scenes = scenes[: total_processes * (len(scenes) // total_processes)]
        else:
            if len(scenes) % total_processes != 0:
                print(
                    "Warning: oversampling some of the scenes to feed all processes."
                    " You can avoid this by setting a number of workers divisor of the number of scenes"
                )
        inds = self._partition_inds(len(scenes), total_processes)

        return {
            "scenes": scenes[inds[process_ind] : inds[process_ind + 1]],
            "max_steps": self.MAX_STEPS,
            "sensors": self.SENSORS,
            "action_space": gym.spaces.Discrete(len(PointNavTask.class_action_names())),
            "seed": seeds[process_ind] if seeds is not None else None,
            "deterministic_cudnn": deterministic_cudnn,
            "rewards_config": {
                "step_penalty": -0.01,
                "goal_success_reward": 10.0,
                "failed_stop_reward": 0.0,
                "shaping_weight": 1.0,  # applied to the decrease in distance to target
            },
        }
