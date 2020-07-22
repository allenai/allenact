import habitat
import random
import time


def speed_demo():
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

    print("Agent stepping around inside environment.")

    count_steps = 0
    ACTIONS = [0, 1, 2, 3]

    start = time.time()
    while count_steps < 10000:
        observations = env.reset()
        while not env.episode_over:
            action = random.choice(ACTIONS)

            observations = env.step(action)
            count_steps += 1
            # cv2.imshow("RGB", transform_rgb_bgr(observations["rgb"]))
    delta = time.time() - start

    print("\n\n\n\n\n\n")
    print("Episode finished after {} steps in {} seconds with {} FPS"
          .format(count_steps, delta, count_steps / delta))
    print("\n\n\n\n\n\n")


if __name__ == "__main__":
    speed_demo()
