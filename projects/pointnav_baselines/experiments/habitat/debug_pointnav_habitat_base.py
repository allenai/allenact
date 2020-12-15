import os
from abc import ABC

import habitat
import torch

from plugins.habitat_plugin.habitat_constants import (
    HABITAT_DATASETS_DIR,
    HABITAT_CONFIGS_DIR,
)
from projects.pointnav_baselines.experiments.habitat.pointnav_habitat_base import PointNavHabitatBaseConfig


class DebugPointNavHabitatBaseConfig(PointNavHabitatBaseConfig, ABC):
    """The base config for all Habitat PointNav experiments."""

    FAILED_END_REWARD = -1.0

    TASK_DATA_DIR_TEMPLATE = os.path.join(
        HABITAT_DATASETS_DIR, "pointnav/habitat-test-scenes/v1/{}/{}.json.gz"
    )
    BASE_CONFIG_YAML_PATH = os.path.join(HABITAT_CONFIGS_DIR, "debug_habitat_pointnav.yaml")

    NUM_TRAIN_PROCESSES = 8 if torch.cuda.is_available() else 4

    TRAIN_GPUS = [torch.cuda.device_count() - 1]
    VALIDATION_GPUS = [torch.cuda.device_count() - 1]
    TESTING_GPUS = [torch.cuda.device_count() - 1]

    @staticmethod
    def make_easy_dataset(dataset: habitat.Dataset) -> habitat.Dataset:
        episodes = [
            e for e in dataset.episodes if float(e.info["geodesic_distance"]) < 1.5
        ]
        for i, e in enumerate(episodes):
            e.episode_id = str(i)
        dataset.episodes = episodes
        return dataset
