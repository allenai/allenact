import os
from abc import ABC

from projects.objectnav_baselines.experiments.objectnav_thor_base import (
    ObjectNavThorBaseConfig,
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
