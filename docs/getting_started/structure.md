# Structure of the codebase

The codebase consists of the following directories: `core`, `docs`, `plugins`, `projects`, `scripts`, `tests` and
`utils`. Below, we explain the overall structure and how different components of the codebase are organized. 

## `core` directory

Contains runtime algorithms for on-policy and off-policy training and inference, base abstractions used throughout
the code base and basic models to be used as building blocks in future models.

* `core.algorithms` includes on-policy and off-policy training nd inference algorithms and abstractions for losses,
policies, rollout storage, etc.

* `core.base_abstractions` includes the base `ExperimentConfig`, distributions, base `Sensor`, `TaskSampler`, `Task`,
etc.

* `core.models` includes basic CNN, and RNN state encoders, besides basic `ActorCriticModel` implementations.

## `docs` directory

Contains documentation for the framework, including guides for installation and first experiments, how-to's for
the definition and usage of different abstractions, tutorials and per-project documentation.

## `plugins` directory

Contains implementations of `ActorCriticModel`s and `Task`s in different environments. Each plugin folder is 
named as `{environment}_plugin` and contains three subfolders:
1. `configs` to host useful configuration for the environment or tasks.
1. `data` to store data to be consumed by the environment or tasks.
1. `scripts` to setup the plugin or gather and process data.

## `projects` directory

Contains project-specific code like experiment configurations and scripts to process results, generate visualizations
or prepare data.

## `scripts` directory

Includes framework-wide scripts to build the documentation, format code, run_tests and start an xserver. The latter can
be used for OpenGL-based environments with super-user privileges in Linux with NVIDIA drivers and `xserver-xorg`
installed.

## `tests` directory

Includes implementations of tests.

## `utils` directory

It includes different types of utilities, mainly divided into:

* `utils.experiment_utils`, including the `TrainingPipeline`, `PipelineStage` and other utilities to configure an
experiment.
* `utils.model_utils`, including generic CNN creation, forward-pass helpers and other utilities.
* `utils.tensor_utils`, including functions to batch observations, convert tensors into video, scale image tensors, etc.
* `utils.viz_utils`, including a `SimpleViz` class that can be instantiated with different visualization plugins during
inference.
* `utils.system`, including logging and networking helpers.

Other utils files, including `utils.misc_utils`, contain a number of helper functions for different purposes.
