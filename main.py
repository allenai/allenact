import sys
import os
from typing import Dict, Tuple

from onpolicy_sync.arguments import get_args
from onpolicy_sync.engine import Trainer, Tester
import inspect
import importlib

from rl_base.experiment_config import ExperimentConfig
from setproctitle import setproctitle as ptitle


def config_source(args) -> Dict[str, Tuple[str, str]]:
    path = os.path.abspath(os.path.normpath(args.experiment_base))
    package = os.path.basename(path)

    module_path = "{}.{}".format(os.path.basename(path), args.experiment)
    modules = [module_path]
    res: Dict[str, Tuple[str, str]] = {}
    while len(modules) > 0:
        new_modules = []
        for module_path in modules:
            if module_path not in res:
                res[module_path] = (os.path.dirname(path), module_path)
                module = importlib.import_module(module_path, package=package)
                for m in inspect.getmembers(module, inspect.isclass):
                    new_module_path = m[1].__module__
                    if new_module_path.split(".")[0] == package:
                        new_modules.append(new_module_path)
        modules = new_modules
    return res


def load_config(args) -> Tuple[ExperimentConfig, Dict[str, Tuple[str, str]]]:
    path = os.path.abspath(os.path.normpath(args.experiment_base))
    sys.path.insert(0, os.path.dirname(path))
    importlib.invalidate_caches()
    module_path = ".{}".format(args.experiment)

    parent = importlib.import_module(os.path.basename(path))
    module = importlib.import_module(module_path, package=os.path.basename(path))

    experiments = [
        m[1]
        for m in inspect.getmembers(module, inspect.isclass)
        if m[1].__module__ == module.__name__
    ]
    assert (
        len(experiments) == 1
    ), "Too many or two few experiments defined in {}".format(module_path)

    config = experiments[0]()
    sources = config_source(args)
    return config, sources


def main():
    args = get_args()
    print("Running with args {}".format(args))

    ptitle("Master: {}".format("Training" if not args.test_date != "" else "Testing"))

    cfg, srcs = load_config(args)

    if args.test_date == "":
        Trainer(
            config=cfg,
            output_dir=args.output_dir,
            loaded_config_src_files=srcs,
            seed=args.seed,
            deterministic_cudnn=args.deterministic_cudnn,
        ).run_pipeline(args.checkpoint)
    else:
        Tester(
            config=cfg,
            output_dir=args.output_dir,
            loaded_config_src_files=srcs,
            seed=args.seed,
            deterministic_cudnn=args.deterministic_cudnn,
        ).run_test(args.test_date, args.checkpoint, args.skip_checkpoints)


if __name__ == "__main__":
    main()
