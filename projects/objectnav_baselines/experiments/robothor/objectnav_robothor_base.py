import os
from abc import ABC
from typing import Any, Dict, List, Optional

import torch
from allenact.utils.misc_utils import prepare_locals_for_super
from projects.objectnav_baselines.experiments.objectnav_thor_base import (
    ObjectNavThorBaseConfig,
)


class ObjectNavRoboThorBaseConfig(ObjectNavThorBaseConfig, ABC):
    """The base config for all RoboTHOR ObjectNav experiments."""

    THOR_COMMIT_ID = "bad5bc2b250615cb766ffb45d455c211329af17e"
    THOR_COMMIT_ID_FOR_RAND_MATERIALS = "9549791ce2e7f472063a10abb1fb7664159fec23"

    AGENT_MODE = "locobot"

    DEFAULT_NUM_TRAIN_PROCESSES = 60 if torch.cuda.is_available() else 1

    TRAIN_DATASET_DIR = os.path.join(os.getcwd(), "datasets/robothor-objectnav/train")
    VAL_DATASET_DIR = os.path.join(os.getcwd(), "datasets/robothor-objectnav/val")
    TEST_DATASET_DIR = os.path.join(os.getcwd(), "datasets/robothor-objectnav/test")

    TARGET_TYPES = tuple(
        sorted(
            [
                "AlarmClock",
                "Apple",
                "BaseballBat",
                "BasketBall",
                "Bed",
                "Bowl",
                "Chair",
                "GarbageCan",
                "HousePlant",
                "Laptop",
                "Mug",
                "Sofa",
                "SprayBottle",
                "Television",
                "Toilet",
                "Vase",
            ]
        )
    )

    def train_task_sampler_args(
        self,
        process_ind: int,
        total_processes: int,
        devices: Optional[List[int]] = None,
        seeds: Optional[List[int]] = None,
        deterministic_cudnn: bool = False,
    ) -> Dict[str, Any]:
        kwargs = super(ObjectNavRoboThorBaseConfig, self).train_task_sampler_args(
            **prepare_locals_for_super(locals())
        )
        if self.randomize_train_materials:
            kwargs["env_args"]["commit_id"] = self.THOR_COMMIT_ID_FOR_RAND_MATERIALS
        return kwargs
