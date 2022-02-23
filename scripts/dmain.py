#!/usr/bin/env python3

"""Entry point to multi-node (distributed) training for a user given experiment
name."""

import os
import random
import string
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

# Add to PYTHONPATH the path of the parent directory of the current file's directory
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(Path(__file__)))))

from allenact.main import get_argument_parser as get_main_arg_parser
from allenact.utils.system import init_logging, get_logger
from constants import ABS_PATH_OF_TOP_LEVEL_DIR


def get_argument_parser():
    """Creates the argument parser."""

    parser = get_main_arg_parser()
    parser.description = f"distributed {parser.description}"

    parser.add_argument(
        "--runs_on",
        required=True,
        type=str,
        help="Comma-separated IP addresses of machines",
    )

    parser.add_argument(
        "--ssh_cmd",
        required=False,
        type=str,
        default="ssh -f {addr}",
        help="SSH command. Useful to utilize a pre-shared key with 'ssh -i mykey.pem -f ubuntu@{addr}'. "
        "The option `-f` should be used for non-interactive session",
    )

    parser.add_argument(
        "--env_activate_path",
        required=True,
        type=str,
        help="Path to the virtual environment's `activate` script. It must be the same across all machines",
    )

    parser.add_argument(
        "--allenact_path",
        required=False,
        type=str,
        default="allenact",
        help="Path to allenact top directory. It must be the same across all machines",
    )

    # Required distributed_ip_and_port
    idx = [a.dest for a in parser._actions].index("distributed_ip_and_port")
    parser._actions[idx].required = True

    return parser


def get_args():
    """Creates the argument parser and parses any input arguments."""

    parser = get_argument_parser()
    args = parser.parse_args()

    return args


def get_raw_args():
    raw_args = sys.argv[1:]
    filtered_args = []
    remove: Optional[str] = None
    enclose_in_quotes: Optional[str] = None
    for arg in raw_args:
        if remove is not None:
            remove = None
        elif enclose_in_quotes is not None:
            # Within backslash expansion: close former single, open double, create single, close double, reopen single
            inner_quote = r"\'\"\'\"\'"
            # Convert double quotes into backslash double for later expansion
            filtered_args.append(
                inner_quote + arg.replace('"', r"\"").replace("'", r"\"") + inner_quote
            )
            enclose_in_quotes = None
        elif arg in [
            "--runs_on",
            "--ssh_cmd",
            "--env_activate_path",
            "--allenact_path",
            "--extra_tag",
            "--machine_id",
        ]:
            remove = arg
        elif arg == "--config_kwargs":
            enclose_in_quotes = arg
            filtered_args.append(arg)
        else:
            filtered_args.append(arg)
    return filtered_args


def wrap_single(text):
    return f"'{text}'"


def wrap_single_nested(text):
    # Close former single, start backslash expansion (via $), create new single quote for expansion:
    quote_enter = r"'$'\'"
    # New closing single quote for expansion, close backslash expansion, reopen former single:
    quote_leave = r"\'''"
    return f"{quote_enter}{text}{quote_leave}"


def wrap_double(text):
    return f'"{text}"'


def id_generator(size=4, chars=string.ascii_uppercase + string.digits):
    return "".join(random.choice(chars) for _ in range(size))


# Assume we can ssh into each of the `runs_on` machines through port 22
if __name__ == "__main__":
    # Tool must be called from AllenAct project's root directory
    cwd = os.path.abspath(os.getcwd())
    assert cwd == ABS_PATH_OF_TOP_LEVEL_DIR, (
        f"`dmain.py` called from {cwd}."
        f"\nIt should be called from AllenAct's top level directory {ABS_PATH_OF_TOP_LEVEL_DIR}."
    )

    args = get_args()

    init_logging(args.log_level)

    raw_args = get_raw_args()

    if args.seed is None:
        seed = random.randint(0, 2 ** 31 - 1)
        raw_args.extend(["-s", f"{seed}"])
        get_logger().info(f"Using random seed {seed} in all workers (none was given)")

    all_addresses = args.runs_on.split(",")
    get_logger().info(f"Running on IP addresses {all_addresses}")

    assert args.distributed_ip_and_port.split(":")[0] in all_addresses, (
        f"Missing listener IP address {args.distributed_ip_and_port.split(':')[0]}"
        f" in list of worker addresses {all_addresses}"
    )

    time_str = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(time.time()))

    global_job_id = id_generator()
    killfilename = os.path.join(
        os.path.expanduser("~"), ".allenact", f"{time_str}_{global_job_id}.killfile"
    )
    os.makedirs(os.path.dirname(killfilename), exist_ok=True)

    code_src = "."

    with open(killfilename, "w") as killfile:
        for it, addr in enumerate(all_addresses):
            code_tget = f"{addr}:{args.allenact_path}/"
            get_logger().info(f"rsync {code_src} to {code_tget}")
            os.system(f"rsync -rz {code_src} {code_tget}")

            job_id = id_generator()

            command = " ".join(
                ["python", "main.py"]
                + raw_args
                + [
                    "--extra_tag",
                    f"{args.extra_tag}{'__' if len(args.extra_tag) > 0 else ''}machine{it}",
                ]
                + ["--machine_id", f"{it}"]
            )

            logfile = (
                f"{args.output_dir}/log_{time_str}_{global_job_id}_{job_id}_machine{it}"
            )

            env_and_command = wrap_single_nested(
                f"for NCCL_SOCKET_IFNAME in $(route | grep default) ; do : ; done && export NCCL_SOCKET_IFNAME"
                f" && cd {args.allenact_path}"
                f" && mkdir -p {args.output_dir}"
                f" && source {args.env_activate_path} &>> {logfile}"
                f" && echo pwd=$(pwd) &>> {logfile}"
                f" && echo output_dir={args.output_dir} &>> {logfile}"
                f" && echo python_version=$(python --version) &>> {logfile}"
                f" && echo python_path=$(which python) &>> {logfile}"
                f" && set | grep NCCL_SOCKET_IFNAME &>> {logfile}"
                f" && echo &>> {logfile}"
                f" && {command} &>> {logfile}"
            )

            screen_name = f"allenact_{time_str}_{global_job_id}_{job_id}_machine{it}"
            screen_command = wrap_single(
                f"screen -S {screen_name} -dm bash -c {env_and_command}"
            )

            ssh_command = f"{args.ssh_cmd.format(addr=addr)} {screen_command}"

            get_logger().debug(f"SSH command {ssh_command}")
            subprocess.run(ssh_command, shell=True, executable="/bin/bash")
            get_logger().info(f"{addr} {screen_name}")

            killfile.write(f"{addr} {screen_name}\n")

    get_logger().info("")
    get_logger().info(f"Running screen ids saved to {killfilename}")
    get_logger().info("")

    get_logger().info("DONE")
