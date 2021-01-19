import os
from abc import ABC

from projects.pointnav_baselines.experiments.pointnav_thor_base import (
    PointNavThorBaseConfig,
)


class PointNaviThorBaseConfig(PointNavThorBaseConfig, ABC):
    """The base config for all iTHOR PointNav experiments."""

    NUM_PROCESSES = 40

    TRAIN_DATASET_DIR = os.path.join(os.getcwd(), "datasets/ithor-pointnav/train")
    VAL_DATASET_DIR = os.path.join(os.getcwd(), "datasets/ithor-pointnav/val")
