from onpolicy_sync.arguments import get_args
from onpolicy_sync.trainer import Trainer
import importlib


def load_config(args):
    module, configclass = args.experiment_config_class.rsplit(".", 1)
    module = importlib.import_module(".".join([args.experiment_base, module]))
    return getattr(module, configclass)()


def main():
    args = get_args()

    Trainer(load_config(args), args.output_dir).run_pipeline(args.checkpoint)


if __name__ == "__main__":
    main()
