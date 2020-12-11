import os
from abc import ABC

from constants import ABS_PATH_OF_TOP_LEVEL_DIR
from projects.objectnav_baselines.experiments.objectnav_thor_base import (
    ObjectNavThorBaseConfig,
)


class ObjectNaviThorBaseConfig(ObjectNavThorBaseConfig, ABC):
    """The base config for all iTHOR ObjectNav experiments."""

    NUM_PROCESSES = 40

    TRAIN_DATASET_DIR = os.path.join(
        ABS_PATH_OF_TOP_LEVEL_DIR, "datasets/ithor-objectnav/train"
    )
    VAL_DATASET_DIR = os.path.join(
        ABS_PATH_OF_TOP_LEVEL_DIR, "datasets/ithor-objectnav/val"
    )

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
            ]
        )
    )
