import os
from abc import ABC

from projects.objectnav_baselines.experiments.objectnav_thor_base import (
    ObjectNavThorBaseConfig,
)

import ai2thor
from packaging import version

if ai2thor.__version__ not in ["0.0.1", None] and version.parse(
    ai2thor.__version__
) < version.parse("2.7.2"):
    raise ImportError(
        "To run the ObjectNavRoboThor baseline experiments you must use"
        " ai2thor version 2.7.1 or higher."
    )


class ObjectNavRoboThorBaseConfig(ObjectNavThorBaseConfig, ABC):
    """The base config for all RoboTHOR ObjectNav experiments."""

    THOR_COMMIT_ID = "bad5bc2b250615cb766ffb45d455c211329af17e"

    NUM_PROCESSES = 60

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
