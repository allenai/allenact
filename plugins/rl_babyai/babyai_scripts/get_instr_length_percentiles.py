import glob
import os

import babyai
import numpy as np

from plugins.rl_babyai.babyai_constants import BABYAI_EXPERT_TRAJECTORIES_DIR

# Boss level
# [(50, 11.0), (90, 22.0), (99, 32.0), (99.9, 38.0), (99.99, 43.0)]

if __name__ == "__main__":
    # level = "BossLevel"
    level = "GoToLocal"
    files = glob.glob(
        os.path.join(BABYAI_EXPERT_TRAJECTORIES_DIR, "*{}-v0.pkl".format(level))
    )
    assert len(files) == 1

    demos = babyai.utils.load_demos(files[0])

    percentiles = [50, 90, 99, 99.9, 99.99, 100]
    print(
        list(
            zip(
                percentiles,
                np.percentile([len(d[0].split(" ")) for d in demos], percentiles),
            )
        )
    )
