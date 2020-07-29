import logging
import socket
import sys
from contextlib import closing

_LOGGER = logging.getLogger("embodiedrl")


class StreamToLogger:
    def __init__(self):
        self.linebuf = ""

    def write(self, buf):
        temp_linebuf = self.linebuf + buf
        self.linebuf = ""
        for line in temp_linebuf.splitlines(True):
            if line[-1] == "\n":
                _LOGGER.info(line.rstrip())
            else:
                self.linebuf += line

    def flush(self):
        if self.linebuf != "":
            _LOGGER.info(self.linebuf.rstrip())
        self.linebuf = ""


def excepthook(*args):
    get_logger().error("Uncaught exception:", exc_info=args)


def get_logger() -> logging.Logger:
    log_format = "default"
    log_level = "debug"

    if len(_LOGGER.handlers) > 0:
        return _LOGGER

    if log_level == "debug":
        log_level = logging.DEBUG
    elif log_level == "info":
        log_level = logging.INFO
    elif log_level == "warning":
        log_level = logging.WARNING
    elif log_level == "error":
        log_level = logging.ERROR
    assert log_level in [
        logging.DEBUG,
        logging.INFO,
        logging.WARNING,
        logging.ERROR,
    ], "unknown log_level {}".format(log_level)

    ch = logging.StreamHandler()
    ch.setLevel(log_level)

    if log_format == "default":
        formatter = logging.Formatter(
            fmt="%(asctime)s: %(levelname)s: %(message)s\t[%(filename)s: %(lineno)d]",
            datefmt="%m/%d %H:%M:%S",
        )
    elif log_format == "defaultMilliseconds":
        formatter = logging.Formatter(
            fmt="%(asctime)s: %(levelname)s: %(message)s\t[%(filename)s: %(lineno)d]"
        )
    else:
        formatter = logging.Formatter(fmt=log_format, datefmt="%m/%d %H:%M:%S")
    ch.setFormatter(formatter)

    _LOGGER.setLevel(log_level)
    _LOGGER.addHandler(ch)

    sys.excepthook = excepthook
    sys.stdout = StreamToLogger()

    return _LOGGER


def find_free_port(address="127.0.0.1"):
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind((address, 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
