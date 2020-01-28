import argparse


def get_args():
    parser = argparse.ArgumentParser(
        description="EmbodiedRL", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "experiment", type=str, help="experiment configuration file name",
    )
    parser.add_argument(
        "output_dir", type=str, help="experiment output folder",
    )

    parser.add_argument(
        "-s", "--seed", required=False, default=None, type=int, help="random seed",
    )
    parser.add_argument(
        "-b",
        "--experiment_base",
        required=False,
        default="experiments",
        type=str,
        help="experiment configuration base folder",
    )
    parser.add_argument(
        "-c",
        "--checkpoint",
        required=False,
        default=None,
        type=str,
        help="optional checkpoint file name to resume training",
    )
    parser.add_argument(
        "-d",
        "--disable_cudnn",
        required=False,
        default=False,
        type=bool,
        help="disables CUDNN",
    )

    return parser.parse_args()
