from typing import cast, Optional, Tuple
from torch import multiprocessing as mp

import logging
import socket
import sys
import io
from contextlib import closing

from constants import ABS_PATH_OF_TOP_LEVEL_DIR

HUMAN_LOG_LEVELS: Tuple[str, ...] = ("debug", "info", "warning", "error", "none")

_LOGGER: Optional[logging.Logger] = None


def get_logger() -> logging.Logger:
    if _new_logger():
        _set_log_formatter()
    return _LOGGER


def init_logging(human_log_level: str = "info"):
    assert human_log_level in HUMAN_LOG_LEVELS, "unknown human_log_level {}".format(
        human_log_level
    )

    if human_log_level == "debug":
        log_level = logging.DEBUG
    elif human_log_level == "info":
        log_level = logging.INFO
    elif human_log_level == "warning":
        log_level = logging.WARNING
    elif human_log_level == "error":
        log_level = logging.ERROR
    elif human_log_level == "none":
        log_level = logging.CRITICAL + 1

    _new_logger(log_level)
    _set_log_formatter()


def find_free_port(address: str = "127.0.0.1") -> int:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind((address, 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def _new_logger(log_level: Optional[int] = None):
    global _LOGGER
    if _LOGGER is None:
        _LOGGER = mp.get_logger()
        if log_level is not None:
            get_logger().setLevel(log_level)
        return True
    return False


def _set_log_formatter():
    assert _LOGGER is not None

    if _LOGGER.getEffectiveLevel() <= logging.CRITICAL:
        default_format = (
            "%(asctime)s %(levelname)s: %(message)s\t[%(filename)s: %(lineno)d]"
        )
        short_date_format = "%m/%d %H:%M:%S"
        log_format = "default"

        ch = logging.StreamHandler()

        if log_format == "default":
            formatter = logging.Formatter(
                fmt=default_format, datefmt=short_date_format,
            )
        elif log_format == "defaultMilliseconds":
            formatter = logging.Formatter(fmt=default_format)
        else:
            formatter = logging.Formatter(fmt=log_format, datefmt=short_date_format)

        ch.setFormatter(formatter)
        ch.addFilter(cast(logging.Filter, _AllenActMessageFilter()))
        _LOGGER.addHandler(ch)

        sys.excepthook = _excepthook
        sys.stdout = cast(io.TextIOWrapper, _StreamToLogger())

    return _LOGGER


class _StreamToLogger:
    def __init__(self):
        self.linebuf = ""

    def write(self, buf):
        temp_linebuf = self.linebuf + buf
        self.linebuf = ""
        for line in temp_linebuf.splitlines(True):
            if line[-1] == "\n":
                cast(logging.Logger, _LOGGER).info(line.rstrip())
            else:
                self.linebuf += line

    def flush(self):
        if self.linebuf != "":
            cast(logging.Logger, _LOGGER).info(self.linebuf.rstrip())
        self.linebuf = ""


def _excepthook(*args):
    get_logger().error("Uncaught exception:", exc_info=args)


class _AllenActMessageFilter:
    def filter(self, record):
        return int(
            ABS_PATH_OF_TOP_LEVEL_DIR in record.pathname or "main" in record.pathname
        )
