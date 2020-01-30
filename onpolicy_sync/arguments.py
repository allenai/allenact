import argparse


def get_args():
    parser = argparse.ArgumentParser(
        description="EmbodiedRL", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "experiment", type=str, help="experiment configuration file name",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        required=False,
        type=str,
        default="experiment_output",
        help='experiment output folder (default: "experiment_output")',
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
        "--deterministic_cudnn",
        dest="deterministic_cudnn",
        action="store_true",
        required=False,
        help="sets CuDNN in deterministic mode",
    )
    parser.set_defaults(deterministic_cudnn=False)

    return parser.parse_args()
