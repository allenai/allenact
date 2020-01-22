from onpolicy_sync.arguments import get_args
from onpolicy_sync.trainer import Trainer
import inspect
import importlib


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

    return experiments[0]()


def main():
    args = get_args()

    Trainer(load_config(args), args.output_dir).run_pipeline(args.checkpoint)


if __name__ == "__main__":
    main()
