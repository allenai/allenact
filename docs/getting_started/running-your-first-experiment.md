# Running your first experiment

## TODO: replace the following with a simple experiment that finishes in 2-3 minutes


Assuming you have [installed all of the requirements](../getting_started/installation.md), you can run your first experiment by calling 

```bash
python ddmain.py object_nav_thor_ppo -s 12345
```
This runs the experiment defined in `experiments/object_nav_thor_ppo.py` with seed 12345.

If everything was installed correctly, a simple semantic navigation model for AI2-THOR will be trained and validated and 
a new folder `experiment_output` will be created containing

* a `checkpoints/LOCAL_TIME_STR/` subfolder with different checkpoints,
* a `used_configs/LOCAL_TIME_STR/` subfolder with all used configuration files,
* and a tensorboard log file under `tb/ObjectNavPPO/LOCAL_TIME_STR/`.

To run your own custom experiment simply define a new experiment configuration in a file `
experiments/my_custom_experiment.py` after which you may run it with
`python ddmain.py my_custom_experiment`.

<!-- ## Experiment configuration

The main entry point for users is a configuration file that defines the experiment we
want to run. More concretely, it includes a single class defining:

* A `tag` to identify the experiment.
* A method to instantiate [actor-critic models](/getting_started/abstractions#actor-critic-model).
* A multi-staged training pipeline with different types of [losses](/getting_started/abstractions#actor-critic-loss), an 
optimizer, and other parameters like learning rates, batch sizes, etc. 
* Machine configuration parameters that will be used e.g. for training or validation.
* A method to instantiate [task samplers](/getting_started/abstractions#task-sampler).
* Methods describing initialization parameters for task samplers used in training, validation, and testing; including
 the assignment of workers to devices for running environments.

A detailed view to an example experiment config file can be found [here](/overview/experiment).
 -->