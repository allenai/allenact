"""Constant values and hyperparameters that are used by the environment."""

import ai2thor.fifo_server


ARM_MIN_HEIGHT = 0.450998873
ARM_MAX_HEIGHT = 1.8009994


ADDITIONAL_ARM_ARGS = {
    "disableRendering": True,
    "returnToStart": True,
    "speed": 1,
}

MOVE_AHEAD = "MoveAheadContinuous"
MOVE_BACK = "MoveBackContinuous"
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
ROTATE_WRIST_PITCH_P = "RotateArmWristPitchP"
ROTATE_WRIST_PITCH_M = "RotateArmWristPitchM"
ROTATE_WRIST_YAW_P = "RotateArmWristYawP"
ROTATE_WRIST_YAW_M = "RotateArmWristYawM"
ROTATE_WRIST_ROLL_P = "RotateArmWristRollP"
ROTATE_WRIST_ROLL_M = "RotateArmWristRollM"
ROTATE_ELBOW_P = "RotateArmElbowP"
ROTATE_ELBOW_M = "RotateArmElbowM"
LOOK_UP = "LookUp"
LOOK_DOWN = "LookDown"
PICKUP = "PickUpMidLevel"
DROP = "DropMidLevel"
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
