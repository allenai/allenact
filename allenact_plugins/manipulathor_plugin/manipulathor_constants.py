"""Constant values and hyperparameters that are used by the environment."""
import json
import os

import ai2thor
import ai2thor.fifo_server
from allenact_plugins.ithor_plugin.ithor_environment import IThorEnvironment
from constants import ABS_PATH_OF_TOP_LEVEL_DIR

MOVE_THR = 0.01
ARM_MIN_HEIGHT = 0.450998873
ARM_MAX_HEIGHT = 1.8009994
MOVE_ARM_CONSTANT = 0.05
MOVE_ARM_HEIGHT_CONSTANT = MOVE_ARM_CONSTANT

ADITIONAL_ARM_ARGS = {
    "disableRendering": True,
    "returnToStart": True,
    "speed": 1,
}

MOVE_AHEAD = "MoveAheadContinuous"
ROTATE_LEFT = "RotateLeftContinuous"
ROTATE_RIGHT = "RotateRightContinuous"
MOVE_ARM_HEIGHT_P = "MoveArmHeightP"
MOVE_ARM_HEIGHT_M = "MoveArmHeightM"
MOVE_ARM_X_P = "MoveArmXP"
MOVE_ARM_X_M = "MoveArmXM"
MOVE_ARM_Y_P = "MoveArmYP"
MOVE_ARM_Y_M = "MoveArmYM"
MOVE_ARM_Z_P = "MoveArmZP"
MOVE_ARM_Z_M = "MoveArmZM"
PICKUP = "PickUpMidLevel"
DONE = "DoneMidLevel"


ENV_ARGS = dict(
    gridSize=0.25,
    width=224,
    height=224,
    visibilityDistance=1.0,
    agentMode="arm",
    fieldOfView=100,
    agentControllerType="mid-level",
    server_class=ai2thor.fifo_server.FifoServer,
    useMassThreshold=True,
    massThreshold=10,
    autoSimulation=False,
    autoSyncTransforms=True,
)

TRAIN_OBJECTS = ["Apple", "Bread", "Tomato", "Lettuce", "Pot", "Mug"]
TEST_OBJECTS = ["Potato", "SoapBottle", "Pan", "Egg", "Spatula", "Cup"]

VALID_OBJECT_LIST = [
    "Knife",
    "Bread",
    "Fork",
    "Potato",
    "SoapBottle",
    "Pan",
    "Plate",
    "Tomato",
    "Egg",
    "Pot",
    "Spatula",
    "Cup",
    "Bowl",
    "SaltShaker",
    "PepperShaker",
    "Lettuce",
    "ButterKnife",
    "Apple",
    "DishSponge",
    "Spoon",
    "Mug",
]

def make_all_objects_unbreakable(controller):
    all_breakable_objects = [
        o["objectType"]
        for o in controller.last_event.metadata["objects"]
        if o["breakable"] is True
    ]
    all_breakable_objects = set(all_breakable_objects)
    for obj_type in all_breakable_objects:
        controller.step(action="MakeObjectsOfTypeUnbreakable", objectType=obj_type)


def reset_environment_and_additional_commands(controller, scene_name):
    controller.reset(scene_name)
    controller.step(action="MakeAllObjectsMoveable")
    controller.step(action="MakeObjectsStaticKinematicMassThreshold")
    make_all_objects_unbreakable(controller)
    return


def transport_wrapper(controller, target_object, target_location):
    transport_detail = dict(
        action="PlaceObjectAtPoint",
        objectId=target_object,
        position=target_location,
        forceKinematic=True,
    )
    advance_detail = dict(action="AdvancePhysicsStep", simSeconds=1.0)

    if issubclass(type(controller), IThorEnvironment):
        event = controller.step(transport_detail)
        controller.step(advance_detail)
    elif type(controller) == ai2thor.controller.Controller:
        event = controller.step(**transport_detail)
        controller.step(**advance_detail)
    return event

dataset_json_file = os.path.join(ABS_PATH_OF_TOP_LEVEL_DIR, "datasets", "apnd-dataset", "starting_pose.json")
try:
    with open(dataset_json_file) as f:
        ARM_START_POSITIONS = json.load(f)
except Exception:
    raise Exception('Dataset not found in {}'.format(dataset_json_file))
