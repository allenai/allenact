# Baseline models for the Point Navigation task in the Habitat, RoboTHOR and iTHOR environments

This project contains the code for training baseline models on the PointNav task. In this setting the agent
spawns at a location in an environment and is tasked to move to another location. The agent is given a "compass"
that tells it the distance and bearing to the target position at every frame. Once the agent is confident that
it has reached the end it executes the `END` action which terminates the episode. If the agent is within a set
distance to the target (in our case 0.2 meters) the agent succeeded, else it failed.

Provided are experiment configs for training a simple convolutional model with
an GRU using `RGB`, `Depth` or `RGBD` as inputs in [Habitat](https://github.com/facebookresearch/habitat-sim), 
[RoboTHOR](https://ai2thor.allenai.org/robothor/) and [iTHOR](https://ai2thor.allenai.org/ithor/).

The experiments are set up to train models using the [DD-PPO](https://arxiv.org/pdf/1911.00357.pdf)
Reinforcement Learning Algorithm.

To train an experiment run the following command from the `allenact` root directory:

```shell script
python main.py -o <PATH_TO_OUTPUT> -c -b <BASE_DIRECTORY_OF_YOUR_EXPERIMENT> <EXPERIMENT_NAME>
```

Where `<PATH_TO_OUTPUT>` is the path of the directory where we want the model weights
and logs to be stored, `<BASE_DIRECTORY_OF_YOUR_EXPERIMENT>` is the directory where our
experiment file is located and `<EXPERIMENT_NAME>` is the name of the python module containing
the experiment. An example usage of this command would be:

```shell script
python main.py -o storage/pointnav-robothor-rgb -b projects/pointnav_baselines/experiments/robothor/ pointnav_robothor_depth_simpleconvgru_ddppo
```

This trains a simple convolutional neural network with a GRU using Depth input on the
PointNav task in the RoboTHOR environment and stores the model weights and logs
to `storage/pointnav-robothor-rgb`.
