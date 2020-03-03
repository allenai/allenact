import habitat
import random
import time
import cv2


FORWARD_KEY="w"
LEFT_KEY="a"
RIGHT_KEY="d"
FINISH="f"


def transform_rgb_bgr(image):
    return image[:, :, [2, 1, 0]]


def agent_demo():
    config = habitat.get_config("habitat/habitat-api/configs/tasks/pointnav.yaml")
    config.defrost()
    config.DATASET.DATA_PATH = "habitat/habitat-api/data/datasets/pointnav/gibson/v1/train/train.json.gz"
    config.DATASET.SCENES_DIR = 'habitat/habitat-api/data/scene_datasets/'
    config.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = 0
    config.freeze()
    env = habitat.Env(
        config=config
    )

    print("Environment creation successful")
    observations = env.reset()
    cv2.imshow("RGB", transform_rgb_bgr(observations["rgb"]))

    print("Agent stepping around inside environment.")

    count_steps = 0
    while not env.episode_over:
        keystroke = cv2.waitKey(0)

        if keystroke == ord(FORWARD_KEY):
            action = 0
            print("action: FORWARD")
        elif keystroke == ord(LEFT_KEY):
            action = 1
            print("action: LEFT")
        elif keystroke == ord(RIGHT_KEY):
            action = 2
            print("action: RIGHT")
        elif keystroke == ord(FINISH):
            action = 3
            print("action: FINISH")
        else:
            print("INVALID KEY")
            continue

        observations = env.step(action)
        count_steps += 1

        print("Position:", env.sim.get_agent_state().position)
        print("Quaternions:", env.sim.get_agent_state().orientation)
        cv2.imshow("RGB", transform_rgb_bgr(observations["rgb"]))

    print("Episode finished after {} steps.".format(count_steps))

    if action == habitat.SimulatorActions.STOP and observations["pointgoal"][0] < 0.2:
        print("you successfully navigated to destination point")
    else:
        print("your navigation was unsuccessful")


if __name__ == "__main__":
    agent_demo()
