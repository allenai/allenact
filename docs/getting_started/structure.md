# Structure of the codebase

The codebase consists of the following directories: `allenact`, `datasets`, `docs`, `overrides`, `allenact_plugins`,
`pretrained_model_ckpts`, `projects`, `scripts`, and `tests`. Below, we explain the overall structure and how
different components of the codebase are organized. 

## [`allenact` directory](https://github.com/allenai/allenact/tree/master/allenact)

Contains runtime algorithms for on-policy and off-policy training and inference, base abstractions used throughout
the code base and basic models to be used as building blocks in future models.

* `allenact.algorithms` includes on-policy and off-policy training nd inference algorithms and abstractions for losses,
policies, rollout storage, etc.

* `allenact.base_abstractions` includes the base `ExperimentConfig`, distributions, base `Sensor`, `TaskSampler`, `Task`,
etc.

* `allenact.embodiedai` includes basic CNN, and RNN state encoders, besides basic `ActorCriticModel` implementations
for embodied AI tasks.

## [`datasets` directory](https://github.com/allenai/allenact/tree/master/datasets)

A directory made to store task-specific datasets. For example, the script `datasets/download_navigation_datasets.sh` can
be used to automatically download task dataset files for Point Navigation within the RoboTHOR environment
and it will place these files into a new `datasets/robothor-pointnav` directory. 

## [`docs` directory](https://github.com/allenai/allenact/tree/master/docs)

Contains documentation for the framework, including guides for installation and first experiments, how-to's for
the definition and usage of different abstractions, tutorials and per-project documentation.

## [`overrides` directory](https://github.com/allenai/allenact/tree/master/overrides)

Files within this directory are used to the look and structure of the documentation generated when running `mkdocs`.
See our [FAQ](../FAQ.md) for information on how to generate this documentation for yourself. 

## [`allenact_plugins` directory](https://github.com/allenai/allenact/tree/master/allenact_plugins)

Contains implementations of `ActorCriticModel`s and `Task`s in different environments. Each plugin folder is 
named as `{environment}_plugin` and contains three subfolders:

1. `configs` to host useful configuration for the environment or tasks.
1. `data` to store data to be consumed by the environment or tasks.
1. `scripts` to setup the plugin or gather and process data.

## [`pretrained_model_ckpts` directory](https://github.com/allenai/allenact/tree/master/pretrained_model_ckpts)

Directory into which pretrained model checkpoints will be saved. See also the 
`pretrained_model_ckpts/download_navigation_model_ckpts.sh` which can be used to download such checkpoints.

## [`projects` directory](https://github.com/allenai/allenact/tree/master/projects)

Contains project-specific code like experiment configurations and scripts to process results, generate visualizations
or prepare data.

## [`scripts` directory](https://github.com/allenai/allenact/tree/master/scripts)

Includes framework-wide scripts to build the documentation, format code, run_tests and start an xserver. The latter can
be used for OpenGL-based environments having super-user privileges in Linux, assuming NVIDIA drivers and `xserver-xorg`
are installed.

## [`tests` directory](https://github.com/allenai/allenact/tree/master/tests)

Includes unit tests for `allenact`.

## [`allenact.utils` directory](https://github.com/allenai/allenact/tree/master/allenact/utils)

It includes different types of utilities, mainly divided into:

* `allenact.utils.experiment_utils`, including the `TrainingPipeline`, `PipelineStage` and other utilities to configure an
experiment.
* `allenact.utils.model_utils`, including generic CNN creation, forward-pass helpers and other utilities.
* `allenact.utils.tensor_utils`, including functions to batch observations, convert tensors into video, scale image tensors, etc.
* `allenact.utils.viz_utils`, including a `VizSuite` class that can be instantiated with different visualization plugins during
inference.
* `allenact.utils.system`, including logging and networking helpers.

Other utils files, including `allenact.utils.misc_utils`, contain a number of helper functions for different purposes.
