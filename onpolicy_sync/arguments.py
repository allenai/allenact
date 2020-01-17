import argparse


def get_args():
    parser = argparse.ArgumentParser(description="EmbodiedRL")
    parser.add_argument(
        "--experiment",
        default="object_nav_thor",
        type=str,
        help="required experiment configuration name",
    )
    parser.add_argument(
        "--output_dir", default="", type=str, help="required experiment output folder",
    )
    parser.add_argument(
        "--checkpoint",
        required=False,
        default=None,
        type=str,
        help="optional checkpoint file name",
    )
    args = parser.parse_args()

    return args
