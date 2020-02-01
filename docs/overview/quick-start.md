# Quick start

Assuming you have [installed all of the requirements](/#installation), you can run your first experiment by calling 

```bash
python main.py object_nav_thor_ppo -s 12345
```
This runs the experiment defined in `experiments/object_nav_thor_ppo.py`.

If everything was installed correctly, a simple semantic navigation model for AI2THOR will be trained and validated and 
a new folder `experiment_output` will be created containing

* a `checkpoints/LOCAL_TIME_STR/` subfolder with different checkpoints,
* a `used_configs` subfolder with all used configuration files,
* and a tensorboard log file under `tb/ObjectNavPPO/LOCAL_TIME_STR/`.

To run your own custom experiment simply define a new experiment configuration in a file `
experiments/my_custom_experiment.py` after which you may run it with
`python main.py my_custom_experiment -s 12345`.

## Experiment configuration

The main entry point for users is a configuration file that defines all the aspects associated with the experiment we
want to run. More concretely, it includes a single class defining:

* A `tag` to identify the experiment.
* A method to instantiate [actor-critic models](/overview/abstractions#actor-critic-model).
* A multi-staged training pipeline with different types of [losses](/overview/abstractions#actor-critic-loss), an 
optimizer and other parameters like learning rates, batch sizes, etc. 
* Machine configuration parameters that will be used e.g. for training or validation.
* A method to instantiate [task samplers](/overview/abstractions#task-sampler).
* Methods describing initialization parameters for task samplers used in training, validation and test.

A detailed view to an example experiment config file can be found [here](/overview/experiment).
