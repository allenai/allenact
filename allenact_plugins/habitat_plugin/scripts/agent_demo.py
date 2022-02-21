import os

import cv2
import habitat
from pyquaternion import Quaternion

from allenact_plugins.habitat_plugin.habitat_constants import (
    HABITAT_CONFIGS_DIR,
    HABITAT_DATASETS_DIR,
    HABITAT_SCENE_DATASETS_DIR,
)
from allenact_plugins.habitat_plugin.habitat_utils import get_habitat_config

FORWARD_KEY = "w"
LEFT_KEY = "a"
RIGHT_KEY = "d"
FINISH = "f"


def transform_rgb_bgr(image):
    return image[:, :, [2, 1, 0]]


def agent_demo():
    config = get_habitat_config(
        os.path.join(HABITAT_CONFIGS_DIR, "tasks/pointnav.yaml")
    )
    config.defrost()
    config.DATASET.DATA_PATH = os.path.join(
        HABITAT_DATASETS_DIR, "pointnav/gibson/v1/train/train.json.gz"
    )
    config.DATASET.SCENES_DIR = HABITAT_SCENE_DATASETS_DIR

    config.DATASET.CONTENT_SCENES = ["Adrian"]

    config.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = 0

    config.freeze()
    env = habitat.Env(config=config)

    print("Environment creation successful")
    observations = env.reset()
    cv2.imshow("RGB", transform_rgb_bgr(observations["rgb"]))

    print("Agent stepping around inside environment.")

    count_steps = 0
    action = None
    while not env.episode_over:
        keystroke = cv2.waitKey(0)

        if keystroke == ord(FORWARD_KEY):
            action = 1
            print("action: FORWARD")
        elif keystroke == ord(LEFT_KEY):
            action = 2
            print("action: LEFT")
        elif keystroke == ord(RIGHT_KEY):
            action = 3
            print("action: RIGHT")
        elif keystroke == ord(FINISH):
            action = 0
            print("action: FINISH")
        else:
            print("INVALID KEY")
            continue

        observations = env.step(action)
        count_steps += 1

        print("Position:", env.sim.get_agent_state().position)
        print("Quaternions:", env.sim.get_agent_state().rotation)
        quat = Quaternion(env.sim.get_agent_state().rotation.components)
        print(quat.radians)
        cv2.imshow("RGB", transform_rgb_bgr(observations["rgb"]))

    print("Episode finished after {} steps.".format(count_steps))

    if action == habitat.SimulatorActions.STOP and observations["pointgoal"][0] < 0.2:
        print("you successfully navigated to destination point")
    else:
        print("your navigation was unsuccessful")


if __name__ == "__main__":
    agent_demo()
