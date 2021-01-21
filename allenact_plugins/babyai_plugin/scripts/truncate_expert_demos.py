import glob
import os

import babyai

from allenact_plugins.babyai_plugin.babyai_constants import (
    BABYAI_EXPERT_TRAJECTORIES_DIR,
)


def make_small_demos(dir: str):
    for file_path in glob.glob(os.path.join(dir, "*.pkl")):
        if "valid" not in file_path and "small" not in file_path:
            new_file_path = file_path.replace(".pkl", "-small.pkl")
            if os.path.exists(new_file_path):
                continue
            print(
                "Saving small version of {} to {}...".format(
                    os.path.basename(file_path), new_file_path
                )
            )
            babyai.utils.save_demos(
                babyai.utils.load_demos(file_path)[:1000], new_file_path
            )
            print("Done.")


if __name__ == "__main__":
    make_small_demos(BABYAI_EXPERT_TRAJECTORIES_DIR)
