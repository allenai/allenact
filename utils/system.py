import logging

LOGGER = logging.getLogger("embodiedrl")


def init_logging(log_format="default", log_level="debug"):
    if len(LOGGER.handlers) > 0:
        return

    if log_level == "debug":
        log_level = logging.DEBUG
    elif log_level == "info":
        log_level = logging.INFO
    elif log_level == "warning":
        log_level = logging.WARNING
    elif log_level == "error":
        log_level = logging.ERROR
    assert log_level in [logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR], \
        "unknown log_level {}".format(log_level)

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

    LOGGER.setLevel(log_level)
    LOGGER.addHandler(ch)
