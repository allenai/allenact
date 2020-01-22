from onpolicy_sync.arguments import get_args
from onpolicy_sync.trainer import Trainer


def main():
    args = get_args()
    if args.experiment == "object_nav_thor":
        from experiments.object_nav_thor import ObjectNavThorExperimentConfig as Config
    elif args.experiment == "object_nav_thor_preresnet":
        from experiments.object_nav_thor_preresnet import (
            ObjectNavThorPreResnetExperimentConfig as Config,
        )
    Trainer(Config(), args.output_dir).run_pipeline(args.checkpoint)


if __name__ == "__main__":
    main()
