#!/usr/bin/env python3

"""Tool to run command on multiple nodes through SSH."""

import argparse
import glob
import os


def get_argument_parser():
    """Creates the argument parser."""

    # noinspection PyTypeChecker
    parser = argparse.ArgumentParser(
        description="dcommand", formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--runs_on",
        required=False,
        type=str,
        default=None,
        help="Comma-separated IP addresses of machines. If empty, the tool will scan for lists of IP addresses"
        " in `screen_ids_file`s in the `~/.allenact` directory.",
    )

    parser.add_argument(
        "--ssh_cmd",
        required=False,
        type=str,
        default="ssh {addr}",
        help="SSH command. Useful to utilize a pre-shared key with 'ssh -i path/to/mykey.pem ubuntu@{addr}'.",
    )

    parser.add_argument(
        "--command",
        required=False,
        default="nvidia-smi | head -n 35",
        type=str,
        help="Command to be run through ssh onto each machine",
    )

    return parser


def get_args():
    """Creates the argument parser and parses any input arguments."""

    parser = get_argument_parser()
    args = parser.parse_args()

    return args


def wrap_double(text):
    return f'"{text}"'


def wrap_single(text):
    return f"'{text}'"


def wrap_single_nested(text, quote=r"'\''"):
    return f"{quote}{text}{quote}"


if __name__ == "__main__":
    args = get_args()

    all_addresses = []
    if args.runs_on is not None:
        all_addresses = args.runs_on.split(",")
    else:
        all_files = sorted(
            glob.glob(os.path.join(os.path.expanduser("~"), ".allenact", "*.killfile")),
            reverse=True,
        )
        if len(all_files) == 0:
            print(
                f"No screen_ids_file found under {os.path.join(os.path.expanduser('~'), '.allenact')}"
            )

        for killfile in all_files:
            with open(killfile, "r") as f:
                # Each line contains 'IP_address screen_ID'
                nodes = [tuple(line[:-1].split(" ")) for line in f.readlines()]

            all_addresses.extend(node[0] for node in nodes)

            use_addresses = ""
            while use_addresses not in ["y", "n"]:
                use_addresses = input(
                    f"Run on {all_addresses} from {killfile}? [Y/n] "
                ).lower()
                if use_addresses == "":
                    use_addresses = "y"

            if use_addresses == "n":
                all_addresses.clear()
            else:
                break

    print(f"Running on IP addresses {all_addresses}")

    for it, addr in enumerate(all_addresses):
        ssh_command = f"{args.ssh_cmd.format(addr=addr)} {wrap_single(args.command)}"

        print(f"{it} {addr} SSH command {ssh_command}")
        os.system(ssh_command)

    print("DONE")
