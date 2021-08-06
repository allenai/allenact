import sys
import os
import time
import random
import string

from allenact.main import get_argument_parser as get_main_arg_parser
from allenact.utils.system import init_logging, get_logger


def get_argument_parser():
    """Creates the argument parser."""

    parser = get_main_arg_parser()

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

    return parser


def get_args():
    """Creates the argument parser and parses any input arguments."""

    parser = get_argument_parser()
    args = parser.parse_args()

    return args


def get_raw_args():
    raw_args = sys.argv[1:]
    filtered_args = []
    remove = None
    for arg in raw_args:
        if remove is not None:
            remove = None
        elif arg in [
            "--runs_on",
            "--ssh_cmd",
            "--env_activate_path",
            "--allenact_path",
        ]:
            remove = arg
        else:
            filtered_args.append(arg)
    return filtered_args


def ws(text):
    return f"'{text}'"


def wd(text):
    return f'"{text}"'


def id_generator(size=4, chars=string.ascii_uppercase + string.digits):
    return "".join(random.choice(chars) for _ in range(size))


# Assume code is deployed in all machines and we can ssh into each of the `runs_on` machines through port 22
# Assume tool is called from allenact project's root directory
if __name__ == "__main__":
    args = get_args()

    init_logging(args.log_level)

    raw_args = get_raw_args()

    all_addresses = args.runs_on.split(",")
    get_logger().info(f"Running on addresses {all_addresses}")

    assert args.distributed_ip_port.split(":")[0] in all_addresses, (
        f"Missing listener IP address {args.distributed_ip_port.split(':')[0]} "
        f"in list of worker addresses {all_addresses}"
    )

    time_str = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(time.time()))

    killfilename = os.path.join(
        os.path.expanduser("~"), ".allenact", f"{time_str}_{id_generator()}.killfile"
    )
    os.makedirs(os.path.dirname(killfilename))

    code_src = "."

    with open(killfilename, "w") as killfile:
        for it, addr in enumerate(all_addresses):
            code_tget = f"{addr}:{args.allenact_path}/"
            get_logger().info(f"rsync {code_src} to {code_tget}")
            os.system(f"rsync -r {code_src} {code_tget}")

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

            logfile = f"{args.output_dir}/log_{time_str}_{job_id}_machine{it}"

            env_and_command = ws(
                f"cd {args.allenact_path} ; "
                f"mkdir -p {args.output_dir} ; "
                f"source {args.env_activate_path} &>> {logfile} ; "
                f"python --version &>> {logfile} ; "
                f"{command} &>> {logfile} & ; "
                f"$! &>> {logfile.replace('log_', 'pid_')} ; "
                f"fg ; "
            )

            screen_name = f"allenact_{time_str}_{job_id}_machine{it}"
            screen_command = wd(
                f"screen -S {screen_name} -dm bash -c {env_and_command}"
            )

            ssh_command = f"{args.ssh_cmd.format(addr=addr)} {screen_command}"

            get_logger().debug(f"SSH command {ssh_command}")
            os.system(ssh_command)
            get_logger().info(f"{addr} {screen_name}")

            killfile.write(f"{addr} {screen_name}\n")

    print(f"Running screen ids saved to {killfilename}")

    print("DONE")
