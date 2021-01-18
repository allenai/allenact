import os

from constants import ABS_PATH_OF_TOP_LEVEL_DIR

if os.path.exists(os.path.join(ABS_PATH_OF_TOP_LEVEL_DIR, "habitat", "habitat-lab")):
    # Old directory structure (not recommended)
    HABITAT_DATA_BASE = os.path.join(
        ABS_PATH_OF_TOP_LEVEL_DIR, "habitat/habitat-lab/data"
    )
else:
    # New directory structure
    HABITAT_DATA_BASE = os.path.join(ABS_PATH_OF_TOP_LEVEL_DIR, "datasets", "habitat",)

HABITAT_DATASETS_DIR = os.path.join(HABITAT_DATA_BASE, "datasets")
HABITAT_SCENE_DATASETS_DIR = os.path.join(HABITAT_DATA_BASE, "scene_datasets")
HABITAT_CONFIGS_DIR = os.path.join(HABITAT_DATA_BASE, "configs")

MOVE_AHEAD = "MOVE_FORWARD"
ROTATE_LEFT = "TURN_LEFT"
ROTATE_RIGHT = "TURN_RIGHT"
LOOK_DOWN = "LOOK_DOWN"
LOOK_UP = "LOOK_UP"
END = "STOP"
