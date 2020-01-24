from onpolicy_sync.arguments import get_args
from onpolicy_sync.trainer import Trainer
import inspect
import importlib


def config_source(args):
    module_path = "{}.{}".format(args.experiment_base, args.experiment)
    valid_folders = {"experiments", args.experiment_base}
    modules, res = [module_path], {}
    while len(modules) > 0:
        new_modules = []
        for module_path in modules:
            module_name = module_path.split(".")[-1]
            if module_name not in res:
                res[module_name] = module_path
                module = importlib.import_module(module_path)
                for m in inspect.getmembers(module, inspect.isclass):
                    new_module_path = m[1].__module__
                    if new_module_path.split(".")[0] in valid_folders:
                        new_modules.append(new_module_path)
        modules = new_modules
    return res


def load_config(args):
    module_path = "{}.{}".format(args.experiment_base, args.experiment)
    module = importlib.import_module(module_path)

    experiments = [
        m[1]
        for m in inspect.getmembers(module, inspect.isclass)
        if m[1].__module__ == module.__name__
    ]
    assert (
        len(experiments) == 1
    ), "Too many or two few experiments defined in {}".format(module_path)

    config = experiments[0]()
    config._loaded_config_src_files = config_source(args)
    return config

def main():
    args = get_args()
    Trainer(load_config(args), args.output_dir).run_pipeline(args.checkpoint)


if __name__ == "__main__":
    main()
