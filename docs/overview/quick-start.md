# Quick start

Assuming you have [installed all of the requirements](/#installation), you can run your first experiment by calling 

```bash
python main.py object_nav_thor runs/object_nav_thor -s 12345
```

If everything was installed correctly, a simple semantic navigation model for AI2THOR will be trained and validated and a new folder `runs/object_nav_thor` will be created containing

* a `models` subfolder with different checkpoints,
* a `used_configs` subfolder with all used configuration files,
* and a tensorboard log file.

# Experiment configuration

The main entry point for users is a configuration file that defines all the aspects associated with the experiment we want to run. More concretely, it includes a single class defining:

* A `tag` to identify the experiment.
* A method to instantiate [actor-critic models](/overview/abstractions#actor-critic-model).
* A multi-staged training pipeline with different types of [losses](/overview/abstractions#actor-critic-loss), an optimizer and other parameters like learning rates, batch sizes, etc. 
* Evaluation parameters that will be used e.g. for validation.
* A method to instantiate [task samplers](/overview/abstractions#task-sampler).
* Methods describing initialization parameters for task samplers used in training, validation and test.

```python

```

