import habitat
import numpy as np
from tqdm import tqdm

map_resolution = 0.05
map_size = 960


def make_map(env, scene):
    vacancy_map = np.zeros([map_size, map_size], dtype=bool)
    for i in tqdm(range(map_size)):
        for j in range(map_size):
            x = (i - map_size//2) * map_resolution
            z = (j - map_size//2) * map_resolution
            vacancy_map[j, i] = env.sim.is_navigable([x, 0.0, z])

    np.save('habitat/habitat-api/data/map_data/pointnav/v1/gibson/data/' + scene, vacancy_map)


def generate_maps():
    config = habitat.get_config("habitat/habitat-api/configs/tasks/pointnav.yaml")
    config.defrost()
    config.DATASET.DATA_PATH = "habitat/habitat-api/data/datasets/pointnav/gibson/v1/train/train.json.gz"
    config.DATASET.SCENES_DIR = 'habitat/habitat-api/data/scene_datasets/'
    config.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = 0
    config.freeze()

    dataset = habitat.make_dataset(config.DATASET.TYPE)
    scenes = dataset.get_scenes_to_load(config.DATASET)

    for scene in scenes:
        print("Making environment for:", scene)
        config.defrost()
        config.DATASET.CONTENT_SCENES = [scene]
        config.freeze()
        env = habitat.Env(
            config=config
        )
        make_map(env, scene)
        env.close()


if __name__ == "__main__":
    generate_maps()
