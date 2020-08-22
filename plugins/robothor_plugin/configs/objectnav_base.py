from math import ceil
from typing import Dict, Any, List, Optional
import abc

import gym
import numpy as np

from core.base_abstractions.experiment_config import ExperimentConfig
from plugins.ithor_plugin.ithor_sensors import RGBSensorThor, GoalObjectTypeThorSensor
from plugins.robothor_plugin.robothor_task_samplers import ObjectNavTaskSampler
from plugins.robothor_plugin.robothor_tasks import ObjectNavTask


class ObjectNavBaseConfig(ExperimentConfig, abc.ABC):
    """An Object Navigation base configuration."""

    # TARGET_TYPES = sorted(
    #     [
    #         "AlarmClock",
    #         "Apple",
    #         "BaseballBat",
    #         "BasketBall",
    #         "Bowl",
    #         "GarbageCan",
    #         "HousePlant",
    #         "Laptop",
    #         "Mug",
    #         # "Remote",  # now it's called RemoteControl, so all epsiodes for this object will be random
    #         "SprayBottle",
    #         "Television",
    #         "Vase",
    #     ]
    # )

    # TARGET_TYPES = sorted(
    #     [
    #         'AlarmClock',
    #         'Apple',
    #         'BasketBall',
    #         'Mug',
    #         'Television',
    #     ]
    # )

    TARGET_TYPES = sorted(["Television", "Mug"])

    CAMERA_WIDTH = 400
    CAMERA_HEIGHT = 300
    SCREEN_SIZE = 224

    VISION_UUID = "rgb"
    TARGET_UUID = "goal_object_type_ind"

    MAX_STEPS = 500

    SENSORS = [
        RGBSensorThor(
            height=SCREEN_SIZE,
            width=SCREEN_SIZE,
            use_resnet_normalization=True,
            uuid=VISION_UUID,
        ),
        GoalObjectTypeThorSensor(object_types=TARGET_TYPES, uuid=TARGET_UUID),
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
    )

    @classmethod
    def make_sampler_fn(cls, **kwargs):
        return ObjectNavTaskSampler(**kwargs)

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
            "object_types": self.TARGET_TYPES,
            "max_steps": self.MAX_STEPS,
            "sensors": self.SENSORS,
            "action_space": gym.spaces.Discrete(
                len(ObjectNavTask.class_action_names())
            ),
            "seed": seeds[process_ind] if seeds is not None else None,
            "deterministic_cudnn": deterministic_cudnn,
            "rewards_config": {
                "step_penalty": -0.01,
                "goal_success_reward": 10.0,
                "failed_stop_reward": 0.0,
                "shaping_weight": 1.0,  # applied to the decrease in distance to target
            },
        }
