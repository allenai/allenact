import os
from abc import ABC

from constants import ABS_PATH_OF_TOP_LEVEL_DIR
from projects.pointnav_baselines.experiments.pointnav_thor_base import (
    PointNavThorBaseConfig,
)


class PointNavRoboThorBaseConfig(PointNavThorBaseConfig, ABC):
    """The base config for all iTHOR PointNav experiments."""

    NUM_PROCESSES = 60

    TRAIN_DATASET_DIR = os.path.join(
        ABS_PATH_OF_TOP_LEVEL_DIR, "datasets/robothor-pointnav/train"
    )
    VAL_DATASET_DIR = os.path.join(
        ABS_PATH_OF_TOP_LEVEL_DIR, "datasets/robothor-pointnav/val"
    )
