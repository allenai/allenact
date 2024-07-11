from abc import ABC
from typing import Dict, Sequence, Optional, List, Any

from allenact.base_abstractions.experiment_config import ExperimentConfig
from allenact.base_abstractions.sensor import Sensor


class GymBaseConfig(ExperimentConfig, ABC):
    SENSORS: Optional[Sequence[Sensor]] = None

    def _get_sampler_args(
        self, process_ind: int, mode: str, seeds: List[int]
    ) -> Dict[str, Any]:
        raise NotImplementedError

    def train_task_sampler_args(
        self,
        process_ind: int,
        total_processes: int,
        devices: Optional[List[int]] = None,
        seeds: Optional[List[int]] = None,
        deterministic_cudnn: bool = False,
    ) -> Dict[str, Any]:
        return self._get_sampler_args(
            process_ind=process_ind, mode="train", seeds=seeds
        )

    def valid_task_sampler_args(
        self,
        process_ind: int,
        total_processes: int,
        devices: Optional[List[int]] = None,
        seeds: Optional[List[int]] = None,
        deterministic_cudnn: bool = False,
    ) -> Dict[str, Any]:
        return self._get_sampler_args(
            process_ind=process_ind, mode="valid", seeds=seeds
        )

    def test_task_sampler_args(
        self,
        process_ind: int,
        total_processes: int,
        devices: Optional[List[int]] = None,
        seeds: Optional[List[int]] = None,
        deterministic_cudnn: bool = False,
    ) -> Dict[str, Any]:
        return self._get_sampler_args(process_ind=process_ind, mode="test", seeds=seeds)
