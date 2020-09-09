import os
import platform
import argparse

from plugins.babyai_plugin.babyai_constants import BABYAI_EXPERT_TRAJECTORIES_DIR

LEVEL_TO_TRAIN_VALID_IDS = {
    "BossLevel": (
        "1DkVVpIEVtpyo1LxOXQL_bVyjFCTO3cHD",
        "1ccEFA_n5RT4SWD0Wa_qO65z2HACJBace",
    ),
    "GoToObjMaze": (
        "1P1CuMbGDJtZit1f-8hmd-HwweXZMj77T",
        "1MVlVsIpJUZ0vjrYGXY6Ku4m4vBxtWjRZ",
    ),
    "GoTo": ("1ABR1q-TClgjSlbhVdVJjzOBpTmTtlTN1", "13DlEx5woi31MIs_dzyLxfi7dPe1g59l2"),
    "GoToLocal": (
        "1U8YWdd3viN2lxOP5BByNUZRPVDKVvDAN",
        "1Esy-J0t8eJUg6_RT8F4kkegHYDWwqmSl",
    ),
}


def get_args():
    """Creates the argument parser and parses input arguments."""

    parser = argparse.ArgumentParser(
        description="download_babyai_expert_demos",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "dataset",
        nargs="?",
        default="all",
        help="dataset name (one of {}, or all)".format(
            ", ".join(LEVEL_TO_TRAIN_VALID_IDS.keys())
        ),
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    if platform.system() == "Linux":
        download_template = """wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id={}' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\\1\\n/p')&id={}" -O {}"""
    elif platform.system() == "Darwin":
        download_template = """wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id={}' -O- | gsed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\\1\\n/p')&id={}" -O {}"""
    else:
        raise NotImplementedError("{} is not supported".format(platform.system()))

    try:
        os.makedirs(BABYAI_EXPERT_TRAJECTORIES_DIR, exist_ok=True)

        if args.dataset == "all":
            id_items = LEVEL_TO_TRAIN_VALID_IDS
        else:
            assert (
                args.dataset in LEVEL_TO_TRAIN_VALID_IDS
            ), "Only {} are valid datasets".format(
                ", ".join(LEVEL_TO_TRAIN_VALID_IDS.keys())
            )
            id_items = {args.dataset: LEVEL_TO_TRAIN_VALID_IDS[args.dataset]}

        for level_name, (train_id, valid_id) in id_items.items():
            train_path = os.path.join(
                BABYAI_EXPERT_TRAJECTORIES_DIR, "BabyAI-{}-v0.pkl".format(level_name)
            )
            if os.path.exists(train_path):
                print("{} already exists, skipping...".format(train_path))
            else:
                os.system(download_template.format(train_id, train_id, train_path))
                print("Demos saved to {}.".format(train_path))

            valid_path = os.path.join(
                BABYAI_EXPERT_TRAJECTORIES_DIR,
                "BabyAI-{}-v0_valid.pkl".format(level_name),
            )
            if os.path.exists(valid_path):
                print("{} already exists, skipping...".format(valid_path))
            else:
                os.system(download_template.format(valid_id, valid_id, valid_path))
                print("Demos saved to {}.".format(valid_path))
    except Exception as _:
        raise Exception(
            "Failed to download babyai demos. Make sure you have the appropriate command line"
            " tools installed for your platform. For MacOS you'll need to install `gsed` and `gwget (the gnu version"
            " of sed) using homebrew or some other method."
        )
