"""Entry point to training/validating/testing for a user given experiment
name."""

import argparse
import glob
import importlib
import inspect
import os
import re
import sys
from typing import Dict, Tuple

import gin
from setproctitle import setproctitle as ptitle

from onpolicy_sync.engine import OnPolicyTrainer, OnPolicyTester
from rl_base.experiment_config import ExperimentConfig
from utils.system import get_logger


def _get_args():
    """Creates the argument parser and parses any input arguments."""

    parser = argparse.ArgumentParser(
        description="EmbodiedRL", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--experiment", type=str, help="experiment configuration file name",
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
        "`skip_checkpoints` or on the single checkpoint saved after a known `test_ckpt_steps` number of steps.",
    )

    parser.add_argument(
        "--env_name",
        required=False,
        type=str,
        default="",
        help="environment name to be sent to any helper scripts (eg. minigrid_random_hp_search)",
    )

    parser.add_argument(
        "--test_ckpt_steps",
        default=None,
        required=False,
        help="when testing, will load the checkpoint with this number of steps.",
    )

    parser.add_argument(
        "-k",
        "--skip_checkpoints",
        required=False,
        default=0,
        type=int,
        help="optional number of skipped checkpoints between runs in test if no checkpoint specified",
    )

    parser.add_argument(
        "--single_process_training",
        action="store_true",
        help="whether or not to train with a single process (useful for debugging).",
    )

    parser.add_argument(
        "--disable_logging",
        action="store_true",
        default=False,
        help="whether or not to disable logging.",
    )

    parser.add_argument(
        "--deterministic_agent",
        action="store_true",
        help="whether or not to train with a single process (useful for debugging).",
    )

    parser.add_argument(
        "--max_training_processes",
        required=False,
        default=None,
        type=int,
        help="maximal number of processes to spawn when training.",
    )

    parser.add_argument(
        "--gp", default=None, action="append", help="values to be used by gin-config.",
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
        if m[1].__module__ == module.__name__ and issubclass(m[1], ExperimentConfig)
    ]
    assert (
        len(experiments) == 1
    ), "Too many or two few experiments defined in {}".format(module_path)

    gin.parse_config_files_and_bindings(None, args.gp)

    config = experiments[0]()
    sources = _config_source(args)
    return config, sources


def find_checkpoint(base_dir, date, steps):
    ckpts = glob.glob(
        os.path.join(
            base_dir, "**", "*time_{}_*steps*{}__seed*.pt".format(date, steps)
        ),
        recursive=True,
    )

    ckpts = [
        ckpt
        for ckpt in ckpts
        if re.match(".*steps_0*{}_.*".format(steps), os.path.basename(ckpt))
    ]
    if len(ckpts) == 0:
        raise FileExistsError(
            "Could not find checkpoint with date {} and {} steps in directory {}.".format(
                date, steps, base_dir
            )
        )
    elif len(ckpts) > 1:
        raise FileExistsError(
            "Too many checkpoints with date {} and {} steps found in directory {}."
            " We found:\n{}".format(date, steps, base_dir, "\n".join(ckpts))
        )

    return ckpts[0]


def main():
    args = _get_args()

    get_logger().info("Running with args {}".format(args))

    ptitle("Master: {}".format("Training" if not args.test_date != "" else "Testing"))

    cfg, srcs = _load_config(args)

    if args.test_date == "":
        trainer = OnPolicyTrainer(
            config=cfg,
            output_dir=args.output_dir,
            loaded_config_src_files=srcs,
            seed=args.seed,
            deterministic_cudnn=args.deterministic_cudnn,
            extra_tag=args.extra_tag,
            single_process_training=args.single_process_training,
            max_training_processes=args.max_training_processes,
        )

        trainer.run_pipeline(
            checkpoint_file_name=args.checkpoint, disable_logging=args.disable_logging
        )
    else:
        checkpoint = args.checkpoint
        if args.test_ckpt_steps is not None:
            assert (
                args.checkpoint is None
            ), "When testing, either specify `checkpoint` or `test_ckpt_steps` but not both."
            checkpoint = find_checkpoint(
                os.path.join(args.output_dir, "checkpoints"),
                date=args.test_date,
                steps=args.test_ckpt_steps,
            )

        test_results = OnPolicyTester(
            config=cfg,
            output_dir=args.output_dir,
            loaded_config_src_files=srcs,
            seed=args.seed,
            deterministic_cudnn=args.deterministic_cudnn,
            single_process_training=args.single_process_training,
            should_log=not args.disable_logging,
        ).run_test(
            experiment_date=args.test_date,
            checkpoint_file_name=checkpoint,
            skip_checkpoints=args.skip_checkpoints,
            deterministic_agent=args.deterministic_agent,
        )

        get_logger().info("Test results: {}".format(test_results))


if __name__ == "__main__":
    main()
