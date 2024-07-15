"""Entry point to training/validating/testing for a user given experiment
name."""

import os

if "CUDA_DEVICE_ORDER" not in os.environ:
    # Necessary to order GPUs correctly in some cases
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import argparse
import ast
import importlib
import inspect
import json
from typing import Dict, List, Optional, Tuple, Type

from setproctitle import setproctitle as ptitle

from allenact import __version__
from allenact.algorithms.onpolicy_sync.runner import (
    CONFIG_KWARGS_STR,
    OnPolicyRunner,
    SaveDirFormat,
)
from allenact.base_abstractions.experiment_config import ExperimentConfig
from allenact.utils.system import HUMAN_LOG_LEVELS, get_logger, init_logging


def get_argument_parser():
    """Creates the argument parser."""

    # noinspection PyTypeChecker
    parser = argparse.ArgumentParser(
        description="allenact",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "experiment",
        type=str,
        help="the path to experiment config file relative the 'experiment_base' directory"
        " (see the `--experiment_base` flag).",
    )

    parser.add_argument(
        "--eval",
        dest="eval",
        action="store_true",
        required=False,
        help="if you pass the `--eval` flag, AllenAct will run inference on your experiment configuration."
        " You will need to specify which experiment checkpoints to run evaluation using the `--checkpoint`"
        " flag.",
    )
    parser.set_defaults(eval=False)

    parser.add_argument(
        "--config_kwargs",
        type=str,
        default=None,
        required=False,
        help="sometimes it is useful to be able to pass additional key-word arguments"
        " to `__init__` when initializing an experiment configuration. This flag can be used"
        " to pass such key-word arugments by specifying them with json, e.g."
        '\n\t--config_kwargs \'{"gpu_id": 0, "my_important_variable": [1,2,3]}\''
        "\nTo see which arguments are supported for your experiment see the experiment"
        " config's `__init__` function. If the value passed to this function is a file path"
        " then we will try to load this file path as a json object and use this json object"
        " as key-word arguments.",
    )

    parser.add_argument(
        "--extra_tag",
        type=str,
        default="",
        required=False,
        help="Add an extra tag to the experiment when trying out new ideas (will be used"
        " as a subdirectory of the tensorboard path so you will be able to"
        " search tensorboard logs using this extra tag). This can also be used to add an extra"
        " organization when running evaluation (e.g. `--extra_tag running_eval_on_great_idea_12`)",
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
        "--save_dir_fmt",
        required=False,
        type=lambda s: SaveDirFormat[s.upper()],
        default="flat",
        help="The file structure to use when saving results from allenact."
        " See documentation o f`SaveDirFormat` for more details."
        " Allowed values are ('flat' and 'nested'). Default: 'flat'.",
    )

    parser.add_argument(
        "-s",
        "--seed",
        required=False,
        default=None,
        type=int,
        help="random seed",
    )
    parser.add_argument(
        "-b",
        "--experiment_base",
        required=False,
        default=os.getcwd(),
        type=str,
        help="experiment configuration base folder (default: working directory)",
    )
    parser.add_argument(
        "-c",
        "--checkpoint",
        required=False,
        default=None,
        type=str,
        help="optional checkpoint file name to resume training on or run testing with. When testing (see the `--eval` flag) this"
        " argument can be used very flexibly as:"
        "\n(1) the path to a particular individual checkpoint file,"
        "\n(2) the path to a directory of checkpoint files all of which you'd like to be evaluated"
        " (checkpoints are expected to have a `.pt` file extension),"
        '\n(3) a "glob" pattern (https://tldp.org/LDP/abs/html/globbingref.html) that will be expanded'
        " using python's `glob.glob` function and should return a collection of checkpoint files."
        "\nIf you'd like to only evaluate a subset of the checkpoints specified by the above directory/glob"
        " (e.g. every checkpoint saved after 5mil steps) you'll likely want to use the `--approx_ckpt_step_interval`"
        " flag.",
    )
    parser.add_argument(
        "--infer_output_dir",
        dest="infer_output_dir",
        action="store_true",
        required=False,
        help="applied when evaluating checkpoint(s) in nested save_dir_fmt: if specified, the output dir will be inferred from checkpoint path.",
    )
    parser.add_argument(
        "--approx_ckpt_step_interval",
        required=False,
        default=None,
        type=float,
        help="if running tests on a collection of checkpoints (see the `--checkpoint` flag) this argument can be"
        " used to skip checkpoints. In particular, if this value is specified and equals `n` then we will"
        " only evaluate checkpoints whose step count is closest to each of `0*n`, `1*n`, `2*n`, `3*n`, ... "
        " n * ceil(max training steps in ckpts / n). Note that 'closest to' is important here as AllenAct does"
        " not generally save checkpoints at exact intervals (doing so would result in performance degregation"
        " in distributed training).",
    )
    parser.add_argument(
        "-r",
        "--restart_pipeline",
        dest="restart_pipeline",
        action="store_true",
        required=False,
        help="for training, if checkpoint is specified, DO NOT continue the training pipeline from where"
        " training had previously ended. Instead restart the training pipeline from scratch but"
        " with the model weights from the checkpoint.",
    )
    parser.set_defaults(restart_pipeline=False)

    parser.add_argument(
        "-d",
        "--deterministic_cudnn",
        dest="deterministic_cudnn",
        action="store_true",
        required=False,
        help="sets CuDNN to deterministic mode",
    )
    parser.set_defaults(deterministic_cudnn=False)

    parser.add_argument(
        "-m",
        "--max_sampler_processes_per_worker",
        required=False,
        default=None,
        type=int,
        help="maximal number of sampler processes to spawn for each worker",
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

    parser.add_argument(
        "-l",
        "--log_level",
        default="info",
        type=str,
        required=False,
        help="sets the log_level. it must be one of {}.".format(
            ", ".join(HUMAN_LOG_LEVELS)
        ),
    )

    parser.add_argument(
        "-i",
        "--disable_tensorboard",
        dest="disable_tensorboard",
        action="store_true",
        required=False,
        help="disable tensorboard logging",
    )
    parser.set_defaults(disable_tensorboard=False)

    parser.add_argument(
        "-a",
        "--disable_config_saving",
        dest="disable_config_saving",
        action="store_true",
        required=False,
        help="disable saving the used config in the output directory",
    )
    parser.set_defaults(disable_config_saving=False)

    parser.add_argument(
        "--collect_valid_results",
        dest="collect_valid_results",
        action="store_true",
        required=False,
        help="enables returning and saving valid results during training",
    )
    parser.set_defaults(collect_valid_results=False)

    parser.add_argument(
        "--valid_on_initial_weights",
        dest="valid_on_initial_weights",
        action="store_true",
        required=False,
        help="enables running validation on the model with initial weights",
    )
    parser.set_defaults(valid_on_initial_weights=False)

    parser.add_argument(
        "--test_expert",
        dest="test_expert",
        action="store_true",
        required=False,
        help="use expert during test",
    )
    parser.set_defaults(test_expert=False)

    parser.add_argument(
        "--version", action="version", version=f"allenact {__version__}"
    )

    parser.add_argument(
        "--distributed_ip_and_port",
        dest="distributed_ip_and_port",
        required=False,
        type=str,
        default="127.0.0.1:0",
        help="IP address and port of listener for distributed process with rank 0."
        " Port number 0 lets runner choose a free port. For more details, please follow the"
        " tutorial https://allenact.org/tutorials/distributed-objectnav-tutorial/.",
    )

    parser.add_argument(
        "--machine_id",
        dest="machine_id",
        required=False,
        type=int,
        default=0,
        help="ID for machine in distributed runs. For more details, please follow the"
        " tutorial https://allenact.org/tutorials/distributed-objectnav-tutorial/",
    )

    parser.add_argument(
        "--save_ckpt_at_every_host",
        dest="save_ckpt_at_every_host",
        action="store_true",
        required=False,
        help="if you pass the `--save_ckpt_at_every_host` flag, AllenAct will save checkpoints at every host as the"
        " the training progresses in distributed training mode.",
    )
    parser.set_defaults(save_ckpt_at_every_host=False)

    parser.add_argument(
        "--callbacks",
        dest="callbacks",
        required=False,
        type=str,
        default="",
        help="Comma-separated list of files with Callback classes to use.",
    )

    parser.add_argument(
        "--enable_crash_recovery",
        dest="enable_crash_recovery",
        default=False,
        action="store_true",
        required=False,
        help="Whether or not to try recovering when a task crashes (use at your own risk).",
    )

    ### DEPRECATED FLAGS
    parser.add_argument(
        "-t",
        "--test_date",
        default=None,
        type=str,
        required=False,
        help="`--test_date` has been deprecated. Please use `--eval` instead.",
    )
    parser.add_argument(
        "--approx_ckpt_steps_count",
        required=False,
        default=None,
        type=float,
        help="`--approx_ckpt_steps_count` has been deprecated."
        " Please specify the checkpoint directly using the '--checkpoint' flag.",
    )
    parser.add_argument(
        "-k",
        "--skip_checkpoints",
        required=False,
        default=0,
        type=int,
        help="`--skip_checkpoints` has been deprecated. Please use `--approx_ckpt_steps_count` instead.",
    )
    ### END DEPRECATED FLAGS

    return parser


def get_args():
    """Creates the argument parser and parses any input arguments."""

    parser = get_argument_parser()
    args = parser.parse_args()

    # check for deprecated
    deprecated_flags = ["test_date", "skip_checkpoints", "approx_ckpt_steps_count"]
    for df in deprecated_flags:
        df_info = parser._option_string_actions[f"--{df}"]
        if getattr(args, df) is not df_info.default:
            raise RuntimeError(df_info.help)

    return args


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
        if not any(isinstance(arg, str) and module_path in arg for arg in e.args):
            raise e
        all_sub_modules = set(find_sub_modules(os.getcwd()))
        desired_config_name = module_path.split(".")[-1]
        relevant_submodules = [
            sm for sm in all_sub_modules if desired_config_name in os.path.basename(sm)
        ]
        raise ModuleNotFoundError(
            f"Could not import experiment '{module_path}', are you sure this is the right path?"
            f" Possibly relevant files include {relevant_submodules}."
            f" Note that the experiment must be reachable along your `PYTHONPATH`, it might"
            f" be helpful for you to run `export PYTHONPATH=$PYTHONPATH:$PWD` in your"
            f" project's top level directory."
        ) from e

    experiments = [
        m[1]
        for m in inspect.getmembers(module, inspect.isclass)
        if m[1].__module__ == module.__name__ and issubclass(m[1], ExperimentConfig)
    ]
    assert (
        len(experiments) == 1
    ), "Too many or two few experiments defined in {}".format(module_path)

    config_kwargs = {}
    if args.config_kwargs is not None:
        if os.path.exists(args.config_kwargs):
            with open(args.config_kwargs, "r") as f:
                config_kwargs = json.load(f)
        else:
            try:
                config_kwargs = json.loads(args.config_kwargs)
            except json.JSONDecodeError:
                get_logger().warning(
                    f"The input for --config_kwargs ('{args.config_kwargs}')"
                    f" does not appear to be valid json. Often this is due to"
                    f" json requiring very specific syntax (e.g. double quoted strings)"
                    f" we'll try to get around this by evaluating with `ast.literal_eval`"
                    f" (a safer version of the standard `eval` function)."
                )
                config_kwargs = ast.literal_eval(args.config_kwargs)

        assert isinstance(
            config_kwargs, Dict
        ), "`--config_kwargs` must be a json string (or a path to a .json file) that evaluates to a dictionary."

    config = experiments[0](**config_kwargs)
    sources = _config_source(config_type=experiments[0])
    sources[CONFIG_KWARGS_STR] = json.dumps(config_kwargs)
    return config, sources


def main():
    args = get_args()

    init_logging(args.log_level)

    get_logger().info("Running with args {}".format(args))

    ptitle("Master: {}".format("Training" if args.eval is None else "Evaluation"))

    cfg, srcs = load_config(args)

    if not args.eval:
        OnPolicyRunner(
            config=cfg,
            output_dir=args.output_dir,
            save_dir_fmt=args.save_dir_fmt,
            loaded_config_src_files=srcs,
            seed=args.seed,
            mode="train",
            deterministic_cudnn=args.deterministic_cudnn,
            deterministic_agents=args.deterministic_agents,
            extra_tag=args.extra_tag,
            disable_tensorboard=args.disable_tensorboard,
            disable_config_saving=args.disable_config_saving,
            distributed_ip_and_port=args.distributed_ip_and_port,
            machine_id=args.machine_id,
            callbacks_paths=args.callbacks,
        ).start_train(
            checkpoint=args.checkpoint,
            restart_pipeline=args.restart_pipeline,
            max_sampler_processes_per_worker=args.max_sampler_processes_per_worker,
            collect_valid_results=args.collect_valid_results,
            valid_on_initial_weights=args.valid_on_initial_weights,
            try_restart_after_task_error=args.enable_crash_recovery,
            save_ckpt_at_every_host=save_ckpt_at_every_host,
        )
    else:
        OnPolicyRunner(
            config=cfg,
            output_dir=args.output_dir,
            save_dir_fmt=args.save_dir_fmt,
            loaded_config_src_files=srcs,
            seed=args.seed,
            mode="test",
            deterministic_cudnn=args.deterministic_cudnn,
            deterministic_agents=args.deterministic_agents,
            extra_tag=args.extra_tag,
            disable_tensorboard=args.disable_tensorboard,
            disable_config_saving=args.disable_config_saving,
            distributed_ip_and_port=args.distributed_ip_and_port,
            machine_id=args.machine_id,
            callbacks_paths=args.callbacks,
        ).start_test(
            checkpoint_path_dir_or_pattern=args.checkpoint,
            infer_output_dir=args.infer_output_dir,
            approx_ckpt_step_interval=args.approx_ckpt_step_interval,
            max_sampler_processes_per_worker=args.max_sampler_processes_per_worker,
            inference_expert=args.test_expert,
        )


if __name__ == "__main__":
    main()
