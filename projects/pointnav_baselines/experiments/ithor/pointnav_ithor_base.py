import os
from abc import ABC

from constants import ABS_PATH_OF_TOP_LEVEL_DIR
from projects.pointnav_baselines.experiments.pointnav_base import PointNavBaseConfig


class PointNaviThorBaseConfig(PointNavBaseConfig, ABC):
    """The base config for all iTHOR PointNav experiments."""

    NUM_PROCESSES = 40

    TRAIN_DATASET_DIR = os.path.join(
        ABS_PATH_OF_TOP_LEVEL_DIR, "datasets/ithor-pointnav/train"
    )
    VAL_DATASET_DIR = os.path.join(
        ABS_PATH_OF_TOP_LEVEL_DIR, "datasets/ithor-pointnav/val"
    )
