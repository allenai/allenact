import argparse


def get_args():
    parser = argparse.ArgumentParser(description="EmbodiedRL")
    parser.add_argument(
        "--experiment_base",
        default="experiments",
        type=str,
        help="experiment configuration base folder, defaults to experiments",
    )
    parser.add_argument(
        "--experiment_config_class",
        default="object_nav_thor.ObjectNavThorExperimentConfig",
        type=str,
        help="experiment configuration class",
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
