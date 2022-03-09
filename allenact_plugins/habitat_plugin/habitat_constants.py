import os

HABITAT_BASE = os.getenv(
    "HABITAT_BASE_DIR",
    default=os.path.join(os.getcwd(), "external_projects", "habitat-lab"),
)
HABITAT_DATA_BASE = os.path.join(os.getcwd(), "data",)

if (not os.path.exists(HABITAT_BASE)) or (not os.path.exists(HABITAT_DATA_BASE)):
    raise ImportError(
        "In order to run properly the Habitat environment makes several assumptions about the file structure of"
        " the local system. The file structure of the current environment does not seem to respect this required"
        " file structure. Please see https://allenact.org/installation/installation-framework/#installation-of-habitat"
        " for details as to how to set up your local environment to make it possible to use the habitat plugin of"
        " AllenAct."
    )

HABITAT_DATASETS_DIR = os.path.join(HABITAT_DATA_BASE, "datasets")
HABITAT_SCENE_DATASETS_DIR = os.path.join(HABITAT_DATA_BASE, "scene_datasets")
HABITAT_CONFIGS_DIR = os.path.join(HABITAT_BASE, "configs")

TESTED_HABITAT_COMMIT = "33654923dc733f5fcea23aea6391034c3f694a67"

MOVE_AHEAD = "MOVE_FORWARD"
ROTATE_LEFT = "TURN_LEFT"
ROTATE_RIGHT = "TURN_RIGHT"
LOOK_DOWN = "LOOK_DOWN"
LOOK_UP = "LOOK_UP"
END = "STOP"
