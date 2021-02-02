import os

from allenact_plugins.robothor_plugin.scripts.make_objectnav_debug_dataset import (
    create_debug_dataset_from_train_dataset,
)

if __name__ == "__main__":
    CURRENT_PATH = os.getcwd()

    SCENE = "FloorPlan1"
    TARGET = "Apple"
    EPISODES = [0, 7, 11, 12]
    BASE_OUT = os.path.join(CURRENT_PATH, "datasets", "ithor-objectnav", "debug")

    create_debug_dataset_from_train_dataset(
        scene=SCENE,
        target_object_type=TARGET,
        episodes_subset=EPISODES,
        train_dataset_path=os.path.join(
            CURRENT_PATH, "datasets", "ithor-objectnav", "train"
        ),
        base_debug_output_path=BASE_OUT,
    )
