# type: ignore
# TODO: @klemenkotar please fix all type errors

import random
from typing import List

import habitat
from habitat import Config


def construct_env_configs(config: Config) -> List[Config]:
    """Create list of Habitat Configs for training on multiple processes To
    allow better performance, dataset are split into small ones for each
    individual env, grouped by scenes.

    # Parameters

    config : configs that contain num_processes as well as information
             necessary to create individual environments.

    # Returns

    List of Configs, one for each process.
    """

    config.freeze()
    num_processes = config.NUM_PROCESSES
    configs = []
    dataset = habitat.make_dataset(config.DATASET.TYPE)
    scenes = dataset.get_scenes_to_load(config.DATASET)

    if len(scenes) > 0:
        random.shuffle(scenes)

        assert len(scenes) >= num_processes, (
            "reduce the number of processes as there " "aren't enough number of scenes"
        )

    scene_splits = [[] for _ in range(num_processes)]
    for idx, scene in enumerate(scenes):
        scene_splits[idx % len(scene_splits)].append(scene)

    assert sum(map(len, scene_splits)) == len(scenes)

    for i in range(num_processes):

        task_config = config.clone()
        task_config.defrost()
        if len(scenes) > 0:
            task_config.DATASET.CONTENT_SCENES = scene_splits[i]

        task_config.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = config.SIMULATOR_GPU_IDS[
            i % len(config.SIMULATOR_GPU_IDS)
        ]

        task_config.freeze()

        configs.append(task_config.clone())

    return configs
