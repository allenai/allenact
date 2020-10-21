import os

import habitat
import numpy as np
from tqdm import tqdm

from plugins.habitat_plugin.habitat_constants import (
    HABITAT_CONFIGS_DIR,
    HABITAT_DATA_BASE,
    HABITAT_SCENE_DATASETS_DIR,
    HABITAT_DATASETS_DIR,
)
from plugins.habitat_plugin.habitat_utils import get_habitat_config

map_resolution = 0.05
map_size = 960


def make_map(env, scene):
    vacancy_map = np.zeros([map_size, map_size], dtype=bool)
    for i in tqdm(range(map_size)):
        for j in range(map_size):
            x = (i - map_size // 2) * map_resolution
            z = (j - map_size // 2) * map_resolution
            vacancy_map[j, i] = env.sim.is_navigable([x, 0.0, z])

    np.save(
        os.path.join(HABITAT_DATA_BASE, "map_data/pointnav/v1/gibson/data/" + scene),
        vacancy_map,
    )


def generate_maps():
    config = get_habitat_config(
        os.path.join(HABITAT_CONFIGS_DIR, "tasks/pointnav.yaml")
    )
    config.defrost()
    config.DATASET.DATA_PATH = os.path.join(
        HABITAT_DATASETS_DIR, "pointnav/gibson/v1/train/train.json.gz"
    )
    config.DATASET.SCENES_DIR = HABITAT_SCENE_DATASETS_DIR
    config.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = 0
    config.freeze()

    dataset = habitat.make_dataset(config.DATASET.TYPE)
    scenes = dataset.get_scenes_to_load(config.DATASET)

    for scene in scenes:
        print("Making environment for:", scene)
        config.defrost()
        config.DATASET.CONTENT_SCENES = [scene]
        config.freeze()
        env = habitat.Env(config=config)
        make_map(env, scene)
        env.close()


if __name__ == "__main__":
    generate_maps()
