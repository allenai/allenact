import os
from abc import ABC

import torch

from projects.objectnav_baselines.experiments.objectnav_thor_base import (
    ObjectNavThorBaseConfig,
)


class ObjectNaviThorBaseConfig(ObjectNavThorBaseConfig, ABC):
    """The base config for all iTHOR ObjectNav experiments."""

    THOR_COMMIT_ID = "9549791ce2e7f472063a10abb1fb7664159fec23"
    AGENT_MODE = "default"

    DEFAULT_NUM_TRAIN_PROCESSES = 40 if torch.cuda.is_available() else 1

    TRAIN_DATASET_DIR = os.path.join(os.getcwd(), "datasets/ithor-objectnav/train")
    VAL_DATASET_DIR = os.path.join(os.getcwd(), "datasets/ithor-objectnav/val")

    TARGET_TYPES = tuple(
        sorted(
            [
                "AlarmClock",
                "Apple",
                "Book",
                "Bowl",
                "Box",
                "Candle",
                "GarbageCan",
                "HousePlant",
                "Laptop",
                "SoapBottle",
                "Television",
                "Toaster",
            ],
        )
    )
