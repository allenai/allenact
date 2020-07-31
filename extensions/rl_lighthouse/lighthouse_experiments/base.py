from typing import Dict, Any, List, Optional, Tuple

import gym
import torch.nn as nn

from extensions.rl_lighthouse.lighthouse_environment import LightHouseEnvironment
from extensions.rl_lighthouse.lighthouse_sensors import FactorialDesignCornerSensor
from extensions.rl_lighthouse.lighthouse_tasks import FindGoalLightHouseTaskSampler
from extensions.rl_lighthouse.lighthouse_util import StopIfNearOptimal
from models.basic_models import LinearActorCritic
from rl_base.experiment_config import ExperimentConfig
from rl_base.sensor import SensorSuite, ExpertPolicySensor, Sensor
from rl_base.task import TaskSampler


class BaseLightHouseExperimentConfig(ExperimentConfig):
    """Base experimental config."""

    WORLD_DIM = 2
    VIEW_RADIUS = 1
    EXPERT_VIEW_RADIUS = 7
    WORLD_RADIUS = 10
    DEGREE = -1
    MAX_STEPS = 1000
    GPU_ID: Optional[int] = None
    NUM_TRAIN_SAMPLERS = 10
    NUM_TEST_TASKS = 100
    _SENSOR_CACHE: Dict[Tuple[int, int, int], List[Sensor]] = {}

    @classmethod
    def get_sensors(cls):
        key = (cls.VIEW_RADIUS, cls.WORLD_DIM, cls.DEGREE, cls.EXPERT_VIEW_RADIUS)
        if key not in cls._SENSOR_CACHE:
            sensors = [
                FactorialDesignCornerSensor(
                    **{
                        "view_radius": cls.VIEW_RADIUS,
                        "world_dim": cls.WORLD_DIM,
                        "degree": cls.DEGREE,
                    }
                )
            ]
            if cls.EXPERT_VIEW_RADIUS:
                sensors.append(
                    ExpertPolicySensor(
                        **{
                            "expert_args": {
                                "expert_view_radius": cls.EXPERT_VIEW_RADIUS
                            },
                            "nactions": 2 * cls.WORLD_DIM,
                        }
                    )
                )
            cls._SENSOR_CACHE[key] = sensors

        return cls._SENSOR_CACHE[key]

    @classmethod
    def optimal_ave_ep_length(cls):
        return LightHouseEnvironment.optimal_ave_ep_length(  # TODO: is this broken?!
            world_dim=cls.WORLD_DIM,
            world_radius=cls.WORLD_RADIUS,
            view_radius=cls.VIEW_RADIUS,
        )

    @classmethod
    def get_early_stopping_criterion(cls):
        optimal_ave_ep_length = cls.optimal_ave_ep_length()

        return StopIfNearOptimal(
            optimal=optimal_ave_ep_length,
            deviation=optimal_ave_ep_length * 0.05,
            min_memory_size=50,
        )

    @classmethod
    def machine_params(cls, mode="train", **kwargs):
        if mode == "train":
            nprocesses = cls.NUM_TRAIN_SAMPLERS
        elif mode == "valid":
            nprocesses = 0
        elif mode == "test":
            nprocesses = 1
        else:
            raise NotImplementedError("mode must be 'train', 'valid', or 'test'.")

        gpu_ids = [] if cls.GPU_ID is None else [cls.GPU_ID]

        return {"nprocesses": nprocesses, "gpu_ids": gpu_ids}

    @classmethod
    def create_model(cls, **kwargs) -> nn.Module:
        sensors = cls.get_sensors()
        return LinearActorCritic(
            input_key=sensors[0]._get_uuid(),
            action_space=gym.spaces.Discrete(2 * cls.WORLD_DIM),
            observation_space=SensorSuite(sensors).observation_spaces,
        )

    @staticmethod
    def make_sampler_fn(**kwargs) -> TaskSampler:
        return FindGoalLightHouseTaskSampler(**kwargs)

    def train_task_sampler_args(
        self,
        process_ind: int,
        total_processes: int,
        devices: Optional[List[int]] = None,
        seeds: Optional[List[int]] = None,
        deterministic_cudnn: bool = False,
    ) -> Dict[str, Any]:
        return {
            "world_dim": self.WORLD_DIM,
            "world_radius": self.WORLD_RADIUS,
            "max_steps": self.MAX_STEPS,
            "sensors": self.get_sensors(),
            "action_space": gym.spaces.Discrete(2 * self.WORLD_DIM),
            "seed": seeds[process_ind] if seeds is not None else None,
        }

    def valid_task_sampler_args(
        self,
        process_ind: int,
        total_processes: int,
        devices: Optional[List[int]] = None,
        seeds: Optional[List[int]] = None,
        deterministic_cudnn: bool = False,
    ) -> Dict[str, Any]:
        raise NotImplementedError

    def test_task_sampler_args(
        self,
        process_ind: int,
        total_processes: int,
        devices: Optional[List[int]] = None,
        seeds: Optional[List[int]] = None,
        deterministic_cudnn: bool = False,
    ) -> Dict[str, Any]:
        return {
            **self.train_task_sampler_args(
                process_ind=process_ind,
                total_processes=total_processes,
                devices=devices,
                seeds=seeds,
                deterministic_cudnn=deterministic_cudnn,
            ),
            "max_tasks": self.NUM_TEST_TASKS,
        }
