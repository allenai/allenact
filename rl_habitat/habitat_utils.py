import random
from typing import List

import habitat
from habitat import Config


def construct_env_configs(config: Config) -> List[Config]:
    r"""Create list of Habitat Configs for training on multiple processes
    To allow better performance, dataset are split into small ones for
    each individual env, grouped by scenes.
    Args:
        config: configs that contain num_processes as well as information
        necessary to create individual environments.
    Returns:
        List of Configs, one for each process
    """

    config.freeze()
    num_processes = config.NUM_PROCESSES
    configs = []
    dataset = habitat.make_dataset(config.DATASET.TYPE)
    # scenes = dataset.get_scenes_to_load(config.DATASET)
    scenes = ['17DRP5sb8fy', '29hnd4uzFmX', '5q7pvUzZiYa', '759xd9YjKW5', '82sE5b5pLXE', '8WUmhLawc2A', 'D7N2EKCX4Sj', 'GdvgFV5R1Z5', 'HxpKQynjfin', 'JF19kD82Mey', 'JeFG25nYj2p', 'Pm6F8kyY3z2', 'S9hNv5qa7GM', 'SN83YJsR3w2', 'Uxmj2M2itWa', 'V2XKFyX4ASd', 'VFuaQ6m2Qom', 'VVfe2KiqLaN', 'XcA2TqTSSAj', 'YmJkqBEsHnH', 'ZMojNkEp431', 'aayBHfsNo7d', 'b8cTxDM8gDG', 'cV4RVeZvu5T', 'e9zR4mvMWw7', 'gZ6f7yhEvPG', 'i5noydFURQK', 'jh4fc5c5qoQ', 'pRbA3pwrgk9', 'qoiz87JEwZ2', 'rPc6DW4iMge', 's8pcmisQ38h']

    if len(scenes) > 0:
        random.shuffle(scenes)

        assert len(scenes) >= num_processes, (
            "reduce the number of processes as there "
            "aren't enough number of scenes"
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

        task_config.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = config.SIMULATOR_GPU_IDS[i % len(config.SIMULATOR_GPU_IDS)]

        task_config.freeze()

        configs.append(task_config.clone())

    return configs

