import os

from allenact_plugins.robothor_plugin.scripts.make_objectnav_debug_dataset import (
    create_debug_dataset_from_train_dataset,
)

if __name__ == "__main__":
    CURRENT_PATH = os.getcwd()
    SCENE = "FloorPlan_Train1_1"
    EPISODES = [3, 4, 5, 6]
    BASE_OUT = os.path.join(CURRENT_PATH, "datasets", "robothor-pointnav", "debug")

    create_debug_dataset_from_train_dataset(
        scene=SCENE,
        target_object_type=None,
        episodes_subset=EPISODES,
        train_dataset_path=os.path.join(
            CURRENT_PATH, "datasets", "robothor-pointnav", "train"
        ),
        base_debug_output_path=BASE_OUT,
    )
