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


class ColoredFormatter(logging.Formatter):
    """Format a log string with colors.

    This implementation taken (with modifications) from
    https://stackoverflow.com/a/384125.
    """

    BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE = range(8)

    RESET_SEQ = "\033[0m"
    COLOR_SEQ = "\033[1;%dm"
    BOLD_SEQ = "\033[1m"

    COLORS = {
        "WARNING": YELLOW,
        "INFO": GREEN,
        "DEBUG": BLUE,
        "ERROR": RED,
        "CRITICAL": MAGENTA,
    }

    def __init__(self, fmt: str, datefmt: Optional[str] = None, use_color=True):
        super().__init__(fmt=fmt, datefmt=datefmt)
        self.use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        levelname = record.levelname
        if self.use_color and levelname in self.COLORS:
            levelname_with_color = (
                self.COLOR_SEQ % (30 + self.COLORS[levelname])
                + levelname
                + self.RESET_SEQ
            )
            record.levelname = levelname_with_color
            formated_record = logging.Formatter.format(self, record)
            record.levelname = (
                levelname  # Resetting levelname as `record` might be used elsewhere
            )
            return formated_record
        else:
            return logging.Formatter.format(self, record)


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
    human_log_level = human_log_level.lower().strip()
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
        add_style_to_logs = True  # In case someone wants to turn this off manually.

        if add_style_to_logs:
            default_format = "$BOLD[%(asctime)s$RESET %(levelname)s$BOLD:]$RESET %(message)s\t[%(filename)s: %(lineno)d]"
            default_format = default_format.replace(
                "$BOLD", ColoredFormatter.BOLD_SEQ
            ).replace("$RESET", ColoredFormatter.RESET_SEQ)
        else:
            default_format = (
                "%(asctime)s %(levelname)s: %(message)s\t[%(filename)s: %(lineno)d]"
            )
        short_date_format = "%m/%d %H:%M:%S"
        log_format = "default"

        if log_format == "default":
            fmt = default_format
            datefmt = short_date_format
        elif log_format == "defaultMilliseconds":
            fmt = default_format
            datefmt = None
        else:
            fmt = log_format
            datefmt = short_date_format

        if add_style_to_logs:
            formatter = ColoredFormatter(fmt=fmt, datefmt=datefmt,)
        else:
            formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)

        ch = logging.StreamHandler()
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
