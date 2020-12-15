import os
from abc import ABC

from constants import ABS_PATH_OF_TOP_LEVEL_DIR
from projects.objectnav_baselines.experiments.objectnav_thor_base import (
    ObjectNavThorBaseConfig,
)


class ObjectNavRoboThorBaseConfig(ObjectNavThorBaseConfig, ABC):
    """The base config for all RoboTHOR ObjectNav experiments."""

    NUM_PROCESSES = 60

    TRAIN_DATASET_DIR = os.path.join(
        ABS_PATH_OF_TOP_LEVEL_DIR, "datasets/robothor-objectnav/train"
    )
    VAL_DATASET_DIR = os.path.join(
        ABS_PATH_OF_TOP_LEVEL_DIR, "datasets/robothor-objectnav/val"
    )

    TARGET_TYPES = tuple(
        sorted(
            [
                "AlarmClock",
                "Apple",
                "BaseballBat",
                "BasketBall",
                "Bowl",
                "GarbageCan",
                "HousePlant",
                "Laptop",
                "Mug",
                "SprayBottle",
                "Television",
                "Vase",
            ]
        )
    )
