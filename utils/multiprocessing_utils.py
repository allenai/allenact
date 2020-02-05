import logging
import logging.handlers
from setproctitle import setproctitle as ptitle
import sys
import traceback

from multiprocessing.managers import BaseManager
import torch.multiprocessing as mp


def logger_loop(
    port,
    log_level=logging.INFO,
    format="%(asctime)s: %(levelname)s: %(message)s \t[%(filename)s: %(lineno)d]",
):
    ptitle("Logger")
    mgr, ctx, port = get_manager(port)

    logger = logging.getLogger("logger")

    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter(fmt=format))
    ch.setLevel(log_level)

    logger.addHandler(ch)
    logger.setLevel(log_level)

    num_records = 0

    queue = mgr.queue("log")
    while True:
        try:
            record = queue.get()
            logger.handle(record)
            num_records += 1
        except EOFError:
            print(
                "Closing logger loop after {} records".format(num_records),
                file=sys.stderr,
            )
            break
        except Exception:
            print("Exception in logger loop:", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            break


def get_manager(
    port=None, ipaddress="127.0.0.1", authkey="embodiedrlauthkey",
):
    server = port is None

    class MPBase(BaseManager):
        pass

    if not server:
        MPBase.register("queue")
        MPBase.register("get_logger")
    else:
        queues = {}

        ctx = mp.get_context("forkserver")  # non-pickle-able

        def queue_impl(name):
            if name not in queues:
                queues[name] = ctx.Queue()
            return queues[name]

        def get_logger(
            logger_name: str, queue_name: str = "log", log_level=logging.INFO
        ) -> logging.Logger:
            logger = logging.getLogger(logger_name)

            if len(logger.handlers) == 0:
                ch = logging.handlers.QueueHandler(queues[queue_name])
                logger.addHandler(ch)
                logger.setLevel(log_level)

            return logger

        MPBase.register("queue", lambda name: queue_impl(name))
        MPBase.register(
            "get_logger",
            lambda logger_name, *args, **kwargs: get_logger(
                logger_name, *args, **kwargs
            ),
        )

    mgr = MPBase(
        address=(ipaddress, port if port is not None else 0), authkey=authkey.encode(),
    )

    if not server:
        mgr.connect()
        ctx = mgr._ctx
    else:
        mgr.start()
        port = mgr._address[1]

        logger = ctx.Process(target=logger_loop, args=(port,))
        logger.daemon = True
        logger.start()

    return mgr, ctx, port
