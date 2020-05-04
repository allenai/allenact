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
    scenes = dataset.get_scenes_to_load(config.DATASET)

    # scenes = ['rPc6DW4iMge', 'e9zR4mvMWw7', 'uNb9QFRL6hY', 'qoiz87JEwZ2', 'sKLMLpTHeUy', 's8pcmisQ38h', '759xd9YjKW5',
    #  '5q7pvUzZiYa', 'XcA2TqTSSAj', 'SN83YJsR3w2', '8WUmhLawc2A', 'JeFG25nYj2p', '17DRP5sb8fy', 'Uxmj2M2itWa',
    #  'D7N2EKCX4Sj', 'b8cTxDM8gDG', 'sT4fr6TAbpF', 'S9hNv5qa7GM', '82sE5b5pLXE', 'pRbA3pwrgk9', 'aayBHfsNo7d',
    #  'cV4RVeZvu5T', 'i5noydFURQK', 'YmJkqBEsHnH', 'jh4fc5c5qoQ', 'VVfe2KiqLaN', '29hnd4uzFmX', 'Pm6F8kyY3z2',
    #  'JF19kD82Mey', 'GdvgFV5R1Z5', 'HxpKQynjfin', 'vyrNrziPKCB']

    scenes = ['29hnd4uzFmX', 'i5noydFURQK', 'cV4RVeZvu5T', '82sE5b5pLXE', 'JeFG25nYj2p', '8WUmhLawc2A', 'VFuaQ6m2Qom',
              'rPc6DW4iMge', '29hnd4uzFmX', 'i5noydFURQK', 'cV4RVeZvu5T', '82sE5b5pLXE',
              'JeFG25nYj2p', '8WUmhLawc2A', 'VFuaQ6m2Qom', 'rPc6DW4iMge', '29hnd4uzFmX', 'i5noydFURQK', 'cV4RVeZvu5T',
              '82sE5b5pLXE', 'JeFG25nYj2p', '8WUmhLawc2A', 'VFuaQ6m2Qom', 'rPc6DW4iMge']

    if len(scenes) > 0:
        # random.shuffle(scenes)

        assert len(scenes) >= num_processes, (
            "reduce the number of processes as there "
            "aren't enough number of scenes"
        )

    scene_splits: List[List] = [[] for _ in range(num_processes)]
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


def construct_env_configs_mp3d(config: Config) -> List[Config]:
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

    if num_processes == 1:
        scene_splits = [['pRbA3pwrgk9']]
    else:
        small = ['rPc6DW4iMge', 'e9zR4mvMWw7', 'uNb9QFRL6hY', 'qoiz87JEwZ2', 'sKLMLpTHeUy', 's8pcmisQ38h', '759xd9YjKW5',
                 '5q7pvUzZiYa', 'XcA2TqTSSAj', 'SN83YJsR3w2', '8WUmhLawc2A', 'JeFG25nYj2p', '17DRP5sb8fy', 'Uxmj2M2itWa',
                 'D7N2EKCX4Sj', 'b8cTxDM8gDG', 'sT4fr6TAbpF', 'S9hNv5qa7GM', '82sE5b5pLXE', 'pRbA3pwrgk9', 'aayBHfsNo7d',
                 'cV4RVeZvu5T', 'i5noydFURQK', 'YmJkqBEsHnH', 'jh4fc5c5qoQ', 'VVfe2KiqLaN', '29hnd4uzFmX', 'Pm6F8kyY3z2',
                 'JF19kD82Mey', 'GdvgFV5R1Z5', 'HxpKQynjfin', 'vyrNrziPKCB']
        med = ['V2XKFyX4ASd', 'VFuaQ6m2Qom', 'ZMojNkEp431', '5LpN3gDmAk7', 'r47D5H71a5s', 'ULsKaCPVFJR',
               'E9uDoFAP3SH', 'kEZ7cmS4wCh', 'ac26ZMwG7aT', 'dhjEzFoUFzH', 'mJXqzFtmKg4', 'p5wJjkQkbXX', 'Vvot9Ly1tCj',
               'EDJbREhghzL', 'VzqfbhrpDEA', '7y3sRwLe3Va']

        scene_splits = [[] for _ in range(config.NUM_PROCESSES)]
        distribute(small, scene_splits, num_gpus=8, procs_per_gpu=3, proc_offset=1, scenes_per_process=2)
        distribute(med, scene_splits, num_gpus=8, procs_per_gpu=3, proc_offset=0, scenes_per_process=1)

        # gpu0 = [['pRbA3pwrgk9', '82sE5b5pLXE', 'S9hNv5qa7GM'],
        #         ['Uxmj2M2itWa', '17DRP5sb8fy', 'JeFG25nYj2p'],
        #         ['5q7pvUzZiYa', '759xd9YjKW5', 's8pcmisQ38h'],
        #         ['e9zR4mvMWw7', 'rPc6DW4iMge', 'vyrNrziPKCB']]
        # gpu1 = [['sT4fr6TAbpF', 'b8cTxDM8gDG', 'D7N2EKCX4Sj'],
        #         ['8WUmhLawc2A', 'SN83YJsR3w2', 'XcA2TqTSSAj'],
        #         ['sKLMLpTHeUy', 'qoiz87JEwZ2', 'uNb9QFRL6hY'],
        #         ['V2XKFyX4ASd', 'VFuaQ6m2Qom', 'ZMojNkEp431']]
        # gpu2 = [['5LpN3gDmAk7', 'r47D5H71a5s', 'ULsKaCPVFJR', 'E9uDoFAP3SH'],
        #         ['VVfe2KiqLaN', 'jh4fc5c5qoQ', 'YmJkqBEsHnH'],  # small
        #         ['i5noydFURQK', 'cV4RVeZvu5T', 'aayBHfsNo7d']]  # small
        # gpu3 = [['kEZ7cmS4wCh', 'ac26ZMwG7aT', 'dhjEzFoUFzH'],
        #         ['mJXqzFtmKg4', 'p5wJjkQkbXX', 'Vvot9Ly1tCj']]
        # gpu4 = [['EDJbREhghzL', 'VzqfbhrpDEA', '7y3sRwLe3Va'],
        #         ['ur6pFq6Qu1A', 'PX4nDJXEHrG', 'PuKPg4mmafe']]
        # gpu5 = [['r1Q1Z4BcV1o', 'gTV8FGcVJC9', '1pXnuDYAj8r'],
        #         ['JF19kD82Mey', 'Pm6F8kyY3z2', '29hnd4uzFmX']]  # small
        # gpu6 = [['VLzqgDo317F', '1LXtFkjw3qL'],
        #         ['HxpKQynjfin', 'gZ6f7yhEvPG', 'GdvgFV5R1Z5']]  # small
        # gpu7 = [['D7G3Y4RVNrH', 'B6ByNegPMKs']]
        #
        # scene_splits = gpu0 + gpu1 + gpu2 + gpu3 + gpu4 + gpu5 + gpu6 + gpu7

    for i in range(num_processes):

        task_config = config.clone()
        task_config.defrost()
        task_config.DATASET.CONTENT_SCENES = scene_splits[i]

        task_config.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = config.SIMULATOR_GPU_IDS[i % len(config.SIMULATOR_GPU_IDS)]

        task_config.freeze()

        configs.append(task_config.clone())

    return configs

def distribute(data: List[str],
               scene_splits: List[List],
               num_gpus=8,
               procs_per_gpu=4,
               proc_offset=0,
               scenes_per_process=1) -> None:
    for idx, scene in enumerate(data):
        i = (idx // num_gpus) % scenes_per_process
        j = idx % num_gpus
        scene_splits[j*procs_per_gpu + i + proc_offset].append(scene)
