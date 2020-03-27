"""Entry point to training/validating/testing for a user given experiment name"""

import argparse
import importlib
import inspect
import logging
import os
import sys
from typing import Dict, Tuple

from setproctitle import setproctitle as ptitle

from onpolicy_sync.runner import OnPolicyRunner
from rl_base.experiment_config import ExperimentConfig
from utils.system import init_logging, LOGGER


def _get_args():
    """Creates the argument parser and parses any input arguments."""

    parser = argparse.ArgumentParser(
        description="EmbodiedRL", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "experiment", type=str, help="experiment configuration file name",
    )

    parser.add_argument(
        "--extra_tag",
        type=str,
        default="",
        required=False,
        help="Add an extra tag to the experiment when trying out new ideas (will be used"
        "as a subdirectory of the tensorboard path so you will be able to"
        "search tensorboard logs using this extra tag).",
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


def _config_source(args) -> Dict[str, Tuple[str, str]]:
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


def _load_config(args) -> Tuple[ExperimentConfig, Dict[str, Tuple[str, str]]]:
    path = os.path.abspath(os.path.normpath(args.experiment_base))
    sys.path.insert(0, os.path.dirname(path))
    importlib.invalidate_caches()
    module_path = ".{}".format(args.experiment)

    importlib.import_module(os.path.basename(path))
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
    sources = _config_source(args)
    return config, sources


def _download_ai2thor():
    from ai2thor.controller import Controller

    try:
        c = Controller(download_only=True, include_private_scenes=True)
        c.stop_unity()
    except Exception as _:
        pass


def main():
    init_logging()

    args = _get_args()

    LOGGER.info("Running with args {}".format(args))

    _download_ai2thor()

    ptitle("Master: {}".format("Training" if not args.test_date != "" else "Testing"))

    cfg, srcs = _load_config(args)

    if args.test_date == "":
        OnPolicyRunner(
            config=cfg,
            output_dir=args.output_dir,
            loaded_config_src_files=srcs,
            seed=args.seed,
            mode="train",
            deterministic_cudnn=args.deterministic_cudnn,
            extra_tag=args.extra_tag,
        ).start_train(args.checkpoint)
    else:
        OnPolicyRunner(
            config=cfg,
            output_dir=args.output_dir,
            loaded_config_src_files=srcs,
            seed=args.seed,
            mode="test",
            deterministic_cudnn=args.deterministic_cudnn,
            extra_tag=args.extra_tag,
        ).start_test(args.test_date, args.checkpoint, args.skip_checkpoints)


if __name__ == "__main__":
    main()
