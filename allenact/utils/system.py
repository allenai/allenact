import io
import logging
import os
import socket
import sys
from contextlib import closing
from typing import cast, Optional, Tuple

from torch import multiprocessing as mp

from allenact._constants import ALLENACT_INSTALL_DIR

HUMAN_LOG_LEVELS: Tuple[str, ...] = ("debug", "info", "warning", "error", "none")
"""
Available log levels: "debug", "info", "warning", "error", "none"
"""

_LOGGER: Optional[logging.Logger] = None


def get_logger() -> logging.Logger:
    """Get a `logging.Logger` to stderr. It can be called whenever we wish to
    log some message. Messages can get mixed-up
    (https://docs.python.org/3.6/library/multiprocessing.html#logging), but it
    works well in most cases.

    # Returns

    logger: the `logging.Logger` object
    """
    if _new_logger():
        if mp.current_process().name == "MainProcess":
            _new_logger(logging.DEBUG)
        _set_log_formatter()
    return _LOGGER


def init_logging(human_log_level: str = "info") -> None:
    """Init the `logging.Logger`.

    It should be called only once in the app (e.g. in `main`). It sets
    the log_level to one of `HUMAN_LOG_LEVELS`. And sets up a handler
    for stderr. The logging level is propagated to all subprocesses.
    """
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
    else:
        raise NotImplementedError(f"Unknown log level {human_log_level}.")

    _new_logger(log_level)
    _set_log_formatter()


def find_free_port(address: str = "127.0.0.1") -> int:
    """Finds a free port for distributed training.

    # Returns

    port: port number that can be used to listen
    """
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind((address, 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        port = s.getsockname()[1]
    return port


def _new_logger(log_level: Optional[int] = None):
    global _LOGGER
    if _LOGGER is None:
        _LOGGER = mp.get_logger()
        if log_level is not None:
            get_logger().setLevel(log_level)
        return True
    if log_level is not None:
        get_logger().setLevel(log_level)
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
        ch.addFilter(cast(logging.Filter, _AllenActMessageFilter(os.getcwd())))
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
    # noinspection PyTypeChecker
    get_logger().error(msg="Uncaught exception:", exc_info=args)


class _AllenActMessageFilter:
    def __init__(self, working_directory: str):
        self.working_directory = working_directory

    # noinspection PyMethodMayBeStatic
    def filter(self, record):
        # TODO: Does this work when pip-installing AllenAct?
        return int(
            self.working_directory in record.pathname
            or ALLENACT_INSTALL_DIR in record.pathname
            or "main" in record.pathname
        )


class ImportChecker:
    def __init__(self, msg=None):
        self.msg = msg

    def __enter__(self):
        pass

    def __exit__(self, exc_type, value, traceback):
        if exc_type == ModuleNotFoundError and self.msg is not None:
            value.msg += self.msg
        return exc_type is None
