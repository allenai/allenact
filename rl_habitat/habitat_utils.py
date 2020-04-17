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
    # dataset = habitat.make_dataset(config.DATASET.TYPE)
    # scenes = dataset.get_scenes_to_load(config.DATASET)
    scenes = ['sT4fr6TAbpF', '29hnd4uzFmX', 'i5noydFURQK', 's8pcmisQ38h', 'rPc6DW4iMge', 'JeFG25nYj2p',
              '82sE5b5pLXE', 'D7N2EKCX4Sj', 'HxpKQynjfin', 'qoiz87JEwZ2', 'aayBHfsNo7d', 'XcA2TqTSSAj',
              '8WUmhLawc2A', 'sKLMLpTHeUy', 'Uxmj2M2itWa', 'Pm6F8kyY3z2', '759xd9YjKW5', 'JF19kD82Mey',
              'V2XKFyX4ASd', '17DRP5sb8fy', '5q7pvUzZiYa', 'VVfe2KiqLaN', 'uNb9QFRL6hY', 'ZMojNkEp431',
              'vyrNrziPKCB', 'e9zR4mvMWw7', 'YmJkqBEsHnH', 'b8cTxDM8gDG', 'GdvgFV5R1Z5', 'pRbA3pwrgk9',
              'jh4fc5c5qoQ', 'S9hNv5qa7GM']

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

