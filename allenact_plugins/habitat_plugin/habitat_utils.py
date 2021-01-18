import glob
import os
import shutil
from typing import List

import habitat
from habitat import Config

from allenact.utils.system import get_logger
from allenact_plugins.habitat_plugin.habitat_constants import (
    HABITAT_DATA_BASE,
    HABITAT_CONFIGS_DIR,
)


def construct_env_configs(
    config: Config, allow_scene_repeat: bool = False,
) -> List[Config]:
    """Create list of Habitat Configs for training on multiple processes To
    allow better performance, dataset are split into small ones for each
    individual env, grouped by scenes.

    # Parameters

    config : configs that contain num_processes as well as information
             necessary to create individual environments.
    allow_scene_repeat: if `True` and the number of distinct scenes
        in the dataset is less than the total number of processes this will
        result in scenes being repeated across processes. If `False`, then
        if the total number of processes is greater than the number of scenes,
        this will result in a RuntimeError exception being raised.

    # Returns

    List of Configs, one for each process.
    """

    config.freeze()
    num_processes = config.NUM_PROCESSES
    configs = []
    dataset = habitat.make_dataset(config.DATASET.TYPE)
    scenes = dataset.get_scenes_to_load(config.DATASET)

    if len(scenes) > 0:
        if len(scenes) < num_processes:
            if not allow_scene_repeat:
                raise RuntimeError(
                    "reduce the number of processes as there aren't enough number of scenes."
                )
            else:
                scenes = (scenes * (1 + (num_processes // len(scenes))))[:num_processes]

    scene_splits: List[List] = [[] for _ in range(num_processes)]
    for idx, scene in enumerate(scenes):
        scene_splits[idx % len(scene_splits)].append(scene)

    assert sum(map(len, scene_splits)) == len(scenes)

    for i in range(num_processes):

        task_config = config.clone()
        task_config.defrost()
        if len(scenes) > 0:
            task_config.DATASET.CONTENT_SCENES = scene_splits[i]

        if len(config.SIMULATOR_GPU_IDS) == 0:
            task_config.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = -1
        else:
            task_config.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = config.SIMULATOR_GPU_IDS[
                i % len(config.SIMULATOR_GPU_IDS)
            ]

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
        scene_splits = [["pRbA3pwrgk9"]]
    else:
        small = [
            "rPc6DW4iMge",
            "e9zR4mvMWw7",
            "uNb9QFRL6hY",
            "qoiz87JEwZ2",
            "sKLMLpTHeUy",
            "s8pcmisQ38h",
            "759xd9YjKW5",
            "XcA2TqTSSAj",
            "SN83YJsR3w2",
            "8WUmhLawc2A",
            "JeFG25nYj2p",
            "17DRP5sb8fy",
            "Uxmj2M2itWa",
            "XcA2TqTSSAj",
            "SN83YJsR3w2",
            "8WUmhLawc2A",
            "JeFG25nYj2p",
            "17DRP5sb8fy",
            "Uxmj2M2itWa",
            "D7N2EKCX4Sj",
            "b8cTxDM8gDG",
            "sT4fr6TAbpF",
            "S9hNv5qa7GM",
            "82sE5b5pLXE",
            "pRbA3pwrgk9",
            "aayBHfsNo7d",
            "cV4RVeZvu5T",
            "i5noydFURQK",
            "YmJkqBEsHnH",
            "jh4fc5c5qoQ",
            "VVfe2KiqLaN",
            "29hnd4uzFmX",
            "Pm6F8kyY3z2",
            "JF19kD82Mey",
            "GdvgFV5R1Z5",
            "HxpKQynjfin",
            "vyrNrziPKCB",
        ]
        med = [
            "V2XKFyX4ASd",
            "VFuaQ6m2Qom",
            "ZMojNkEp431",
            "5LpN3gDmAk7",
            "r47D5H71a5s",
            "ULsKaCPVFJR",
            "E9uDoFAP3SH",
            "kEZ7cmS4wCh",
            "ac26ZMwG7aT",
            "dhjEzFoUFzH",
            "mJXqzFtmKg4",
            "p5wJjkQkbXX",
            "Vvot9Ly1tCj",
            "EDJbREhghzL",
            "VzqfbhrpDEA",
            "7y3sRwLe3Va",
        ]

        scene_splits = [[] for _ in range(config.NUM_PROCESSES)]
        distribute(
            small,
            scene_splits,
            num_gpus=8,
            procs_per_gpu=3,
            proc_offset=1,
            scenes_per_process=2,
        )
        distribute(
            med,
            scene_splits,
            num_gpus=8,
            procs_per_gpu=3,
            proc_offset=0,
            scenes_per_process=1,
        )

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

        task_config.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = config.SIMULATOR_GPU_IDS[
            i % len(config.SIMULATOR_GPU_IDS)
        ]

        task_config.freeze()

        configs.append(task_config.clone())

    return configs


def distribute(
    data: List[str],
    scene_splits: List[List],
    num_gpus=8,
    procs_per_gpu=4,
    proc_offset=0,
    scenes_per_process=1,
) -> None:
    for idx, scene in enumerate(data):
        i = (idx // num_gpus) % scenes_per_process
        j = idx % num_gpus
        scene_splits[j * procs_per_gpu + i + proc_offset].append(scene)


def get_habitat_config(path: str, allow_download: bool = True):
    assert (
        path[-4:].lower() == ".yml" or path[-5:].lower() == ".yaml"
    ), f"path ({path}) must be a .yml or .yaml file."

    if not os.path.exists(path):
        if not allow_download:
            raise IOError(
                "Path {} does not exist and we do not wish to try downloading it."
            )

        get_logger().info(
            f"Attempting to load config at path {path}. This path does not exist, attempting to"
            f"download habitat configs and will try again. Downloading..."
        )

        os.chdir(HABITAT_DATA_BASE)

        output_archive_name = "__TO_OVERWRITE__.zip"
        deletable_dir_name = "__TO_DELETE__"

        url = "https://github.com/facebookresearch/habitat-lab/archive/7c4286653211bbfaca59d0807c28bfb3a6b962bf.zip"
        cmd = f"wget {url} -O {output_archive_name}"
        if os.system(cmd):
            raise RuntimeError(f"ERROR: `{cmd}` failed.")

        cmd = f"unzip {output_archive_name} -d {deletable_dir_name}"
        if os.system(cmd):
            raise RuntimeError(f"ERROR: `{cmd}` failed.")

        habitat_path = glob.glob(os.path.join(deletable_dir_name, "habitat-lab*"))[0]
        cmd = f"rsync --ignore-existing -raz {habitat_path}/configs/ {HABITAT_CONFIGS_DIR}/"
        if os.system(cmd):
            raise RuntimeError(f"ERROR: `{cmd}` failed.")

        os.remove(output_archive_name)
        shutil.rmtree(deletable_dir_name)

        if not os.path.exists(path):
            raise RuntimeError(
                f"Config at path {path} does not exist even after downloading habitat configs to {HABITAT_CONFIGS_DIR}."
            )
        else:
            get_logger().info(f"Config downloaded successfully.")

    return habitat.get_config(path)
