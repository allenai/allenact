#!/usr/bin/env python3

import os
import argparse


def get_argument_parser():
    """Creates the argument parser."""

    # noinspection PyTypeChecker
    parser = argparse.ArgumentParser(
        description="dcommand", formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

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

    all_addresses = args.runs_on.split(",")
    print(f"Running on addresses {all_addresses}")

    for it, addr in enumerate(all_addresses):
        ssh_command = f"{args.ssh_cmd.format(addr=addr)} {wrap_single(args.command)}"

        print(f"{it} {addr} SSH command {ssh_command}")
        os.system(ssh_command)

    print("DONE")
