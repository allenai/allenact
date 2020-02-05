import time
import random
import sys
import traceback
import copy

from utils.multiprocessing_utils import get_manager


def children_loop(port, id):
    mgr, ctx, port = get_manager(port)

    parent_children_queue = mgr.queue("parent_children")
    logger = mgr.get_logger("children")
    while True:
        try:
            data = parent_children_queue.get()
            if isinstance(data, tuple) and copy.copy(data[0]) == "parent":
                for it in range(2):
                    message = "{}".format(("child", id, it, data))
                    logger.info(message)
                    time.sleep(0.1 + 0.2 * random.random())
            else:
                parent_children_queue.put(data)
        except EOFError:
            print("Closing child loop", file=sys.stderr)
            break
        except Exception:
            print("Exception in child loop:", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            break


def parent_loop(port, id):
    mgr, ctx, port = get_manager(port)

    parent_children_queue = mgr.queue("parent_children")
    workers = []
    for it in range(1):
        workers.append(ctx.Process(target=children_loop, args=(port, it,)))
        workers[-1].daemon = True
        workers[-1].start()

    logger = mgr.get_logger("parent")
    for it in range(5):
        logger.info("{}".format(("parent", id, it)))
        parent_children_queue.put(("parent", id, it))
        time.sleep(0.5 + 1.0 * random.random())


if __name__ == "__main__":
    mgr, ctx, port = get_manager(port=None)  # manager as server

    workers = []
    for it in range(1):
        workers.append(ctx.Process(target=parent_loop, args=(port, it,)))
        workers[-1].start()

    time.sleep(10)

    for worker in workers:
        worker.join()
