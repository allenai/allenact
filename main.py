import sys
import os
from typing import Dict, Tuple
import argparse
import inspect
import importlib
import logging

from setproctitle import setproctitle as ptitle

from onpolicy_sync.engine import Trainer, Tester
from rl_base.experiment_config import ExperimentConfig

logger = logging.getLogger("embodiedrl")

"""Entry point to training/validating/testing for a user given experiment name"""


def get_args():
    """Creates the argument parser and parses any input arguments."""

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
        help="experiment output folder",
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

    parser.add_argument(
        "-t",
        "--test_date",
        default="",
        required=False,
        help="tests the experiment run on specified date (formatted as %%Y-%%m-%%d_%%H-%%M-%%S), assuming it was "
        "previously trained. If no checkpoint is specified, it will run on all checkpoints enabled by "
        "skip_checkpoints",
    )

    parser.add_argument(
        "-k",
        "--skip_checkpoints",
        required=False,
        default=0,
        type=int,
        help="optional number of skipped checkpoints between runs in test if no checkpoint specified",
    )

    return parser.parse_args()


def config_source(args) -> Dict[str, Tuple[str, str]]:
    path = os.path.abspath(os.path.normpath(args.experiment_base))
    package = os.path.basename(path)

    module_path = "{}.{}".format(os.path.basename(path), args.experiment)
    modules = [module_path]
    res: Dict[str, Tuple[str, str]] = {}
    while len(modules) > 0:
        new_modules = []
        for module_path in modules:
            if module_path not in res:
                res[module_path] = (os.path.dirname(path), module_path)
                module = importlib.import_module(module_path, package=package)
                for m in inspect.getmembers(module, inspect.isclass):
                    new_module_path = m[1].__module__
                    if new_module_path.split(".")[0] == package:
                        new_modules.append(new_module_path)
        modules = new_modules
    return res


def load_config(args) -> Tuple[ExperimentConfig, Dict[str, Tuple[str, str]]]:
    path = os.path.abspath(os.path.normpath(args.experiment_base))
    sys.path.insert(0, os.path.dirname(path))
    importlib.invalidate_caches()
    module_path = ".{}".format(args.experiment)

    parent = importlib.import_module(os.path.basename(path))
    module = importlib.import_module(module_path, package=os.path.basename(path))

    experiments = [
        m[1]
        for m in inspect.getmembers(module, inspect.isclass)
        if m[1].__module__ == module.__name__
    ]
    assert (
        len(experiments) == 1
    ), "Too many or two few experiments defined in {}".format(module_path)

    config = experiments[0]()
    sources = config_source(args)
    return config, sources


def init_logging(log_format="default", log_level="debug"):
    if log_level == "debug":
        base_logging_level = logging.DEBUG
    elif log_level == "info":
        base_logging_level = logging.INFO
    elif log_level == "warning":
        base_logging_level = logging.WARNING
    else:
        raise TypeError("%s is an incorrect logging type!", log_level)
    if len(logger.handlers) == 0:
        ch = logging.StreamHandler()
        logger.setLevel(base_logging_level)
        ch.setLevel(base_logging_level)
        if log_format == "default":
            formatter = logging.Formatter(
                fmt="%(asctime)s: %(levelname)s: %(message)s \t[%(filename)s: %(lineno)d]",
                datefmt="%m/%d %I:%M:%S",
            )
        elif log_format == "defaultMilliseconds":
            formatter = logging.Formatter(
                fmt="%(asctime)s: %(levelname)s: %(message)s \t[%(filename)s: %(lineno)d]"
            )
        else:
            formatter = logging.Formatter(fmt=log_format, datefmt="%m/%d %I:%M:%S")

        ch.setFormatter(formatter)
        logger.addHandler(ch)


def download_ai2thor():
    from ai2thor.controller import Controller

    Controller(download_only=True)


def main():
    init_logging()

    args = get_args()

    logger.info("Running with args {}".format(args))

    download_ai2thor()

    ptitle("Master: {}".format("Training" if not args.test_date != "" else "Testing"))

    cfg, srcs = load_config(args)

    if args.test_date == "":
        Trainer(
            config=cfg,
            output_dir=args.output_dir,
            loaded_config_src_files=srcs,
            seed=args.seed,
            deterministic_cudnn=args.deterministic_cudnn,
        ).run_pipeline(args.checkpoint)
    else:
        Tester(
            config=cfg,
            output_dir=args.output_dir,
            loaded_config_src_files=srcs,
            seed=args.seed,
            deterministic_cudnn=args.deterministic_cudnn,
        ).run_test(args.test_date, args.checkpoint, args.skip_checkpoints)


if __name__ == "__main__":
    main()
