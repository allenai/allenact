"""Entry point to training/validating/testing for a user given experiment
name."""

import argparse
import importlib
import inspect
import os
from typing import Dict, Tuple, List, Optional, Type

import gin
from setproctitle import setproctitle as ptitle

from constants import ABS_PATH_OF_TOP_LEVEL_DIR
from core.algorithms.onpolicy_sync.runner import OnPolicyRunner
from core.base_abstractions.experiment_config import ExperimentConfig
from utils.system import get_logger


def get_args():
    """Creates the argument parser and parses any input arguments."""

    parser = argparse.ArgumentParser(
        description="allenact", formatter_class=argparse.ArgumentDefaultsHelpFormatter,
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
        help="optional checkpoint file name to resume training or test",
    )
    parser.add_argument(
        "-r",
        "--restart_pipeline",
        dest="restart_pipeline",
        action="store_true",
        required=False,
        help="for training, if checkpoint is specified, DO NOT continue the training pipeline from where"
        "training had previously ended. Instead restart the training pipeline from scratch but"
        "with the model weights from the checkpoint.",
    )
    parser.set_defaults(restart_pipeline=False)

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
        default=None,
        type=str,
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

    parser.add_argument(
        "-m",
        "--max_sampler_processes_per_worker",
        required=False,
        default=None,
        type=int,
        help="maximal number of sampler processes to spawn for each worker",
    )

    parser.add_argument(
        "--gp", default=None, action="append", help="values to be used by gin-config.",
    )

    parser.add_argument(
        "-e",
        "--deterministic_agents",
        dest="deterministic_agents",
        action="store_true",
        required=False,
        help="enable deterministic agents (i.e. always taking the mode action) during validation/testing",
    )
    parser.set_defaults(deterministic_agents=False)

    return parser.parse_args()


def _config_source(config_type: Type) -> Dict[str, str]:
    if config_type is ExperimentConfig:
        return {}

    try:
        module_file_path = inspect.getfile(config_type)
        module_dot_path = config_type.__module__
        sources_dict = {module_file_path: module_dot_path}
        for super_type in config_type.__bases__:
            sources_dict.update(_config_source(super_type))

        return sources_dict
    except TypeError as _:
        return {}


def find_sub_modules(path: str, module_list: Optional[List] = None):
    if module_list is None:
        module_list = []

    path = os.path.abspath(path)
    if path[-3:] == ".py":
        module_list.append(path)
    elif os.path.isdir(path):
        contents = os.listdir(path)
        if any(key in contents for key in ["__init__.py", "setup.py"]):
            new_paths = [os.path.join(path, f) for f in os.listdir(path)]
            for new_path in new_paths:
                find_sub_modules(new_path, module_list)
    return module_list


def load_config(args) -> Tuple[ExperimentConfig, Dict[str, str]]:
    assert os.path.exists(
        args.experiment_base
    ), "The path '{}' does not seem to exist (your current working directory is '{}').".format(
        args.experiment_base, os.getcwd()
    )
    rel_base_dir = os.path.relpath(  # Normalizing string representation of path
        os.path.abspath(args.experiment_base), os.getcwd()
    )
    rel_base_dot_path = rel_base_dir.replace("/", ".")
    if rel_base_dot_path == ".":
        rel_base_dot_path = ""

    exp_dot_path = args.experiment
    if exp_dot_path[-3:] == ".py":
        exp_dot_path = exp_dot_path[:-3]
    exp_dot_path = exp_dot_path.replace("/", ".")

    module_path = (
        f"{rel_base_dot_path}.{exp_dot_path}"
        if len(rel_base_dot_path) != 0
        else exp_dot_path
    )

    try:
        importlib.invalidate_caches()
        module = importlib.import_module(module_path)
    except ModuleNotFoundError as e:
        all_sub_modules = set(find_sub_modules(os.getcwd())) | set(
            find_sub_modules(ABS_PATH_OF_TOP_LEVEL_DIR)
        )
        desired_config_name = module_path.split(".")[-1]
        relevant_submodules = [
            sm for sm in all_sub_modules if desired_config_name in os.path.basename(sm)
        ]
        raise ModuleNotFoundError(
            "Could not import experiment '{}', are you sure this is the right path?"
            " Possibly relevant files include {}.".format(
                module_path, relevant_submodules
            ),
        ) from e

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
    sources = _config_source(config_type=experiments[0])
    return config, sources


def main():
    args = get_args()

    get_logger().info("Running with args {}".format(args))

    ptitle("Master: {}".format("Training" if args.test_date is None else "Testing"))

    cfg, srcs = load_config(args)

    if args.test_date is None:
        OnPolicyRunner(
            config=cfg,
            output_dir=args.output_dir,
            loaded_config_src_files=srcs,
            seed=args.seed,
            mode="train",
            deterministic_cudnn=args.deterministic_cudnn,
            deterministic_agents=args.deterministic_agents,
            extra_tag=args.extra_tag,
        ).start_train(
            checkpoint=args.checkpoint,
            restart_pipeline=args.restart_pipeline,
            max_sampler_processes_per_worker=args.max_sampler_processes_per_worker,
        )
    else:
        OnPolicyRunner(
            config=cfg,
            output_dir=args.output_dir,
            loaded_config_src_files=srcs,
            seed=args.seed,
            mode="test",
            deterministic_cudnn=args.deterministic_cudnn,
            deterministic_agents=args.deterministic_agents,
            extra_tag=args.extra_tag,
        ).start_test(
            experiment_date=args.test_date,
            checkpoint=args.checkpoint,
            skip_checkpoints=args.skip_checkpoints,
            max_sampler_processes_per_worker=args.max_sampler_processes_per_worker,
        )


if __name__ == "__main__":
    main()
