import json
import os

from constants import ABS_PATH_OF_TOP_LEVEL_DIR

TRAIN_OBJECTS = ["Apple", "Bread", "Tomato", "Lettuce", "Pot", "Mug"]
TEST_OBJECTS = ["Potato", "SoapBottle", "Pan", "Egg", "Spatula", "Cup"]
MOVE_ARM_CONSTANT = 0.05
MOVE_ARM_HEIGHT_CONSTANT = MOVE_ARM_CONSTANT
UNWANTED_MOVE_THR = 0.01
DISTANCE_EPS = 1e-9
DISTANCE_MAX = 10.0

dataset_json_file = os.path.join(
    ABS_PATH_OF_TOP_LEVEL_DIR, "datasets", "apnd-dataset", "starting_pose.json"
)
try:
    with open(dataset_json_file) as f:
        ARM_START_POSITIONS = json.load(f)
except Exception:
    raise Exception("Dataset not found in {}".format(dataset_json_file))
