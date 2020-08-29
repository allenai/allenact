# Baseline models for the Object Navigation task in the RoboTHOR and iTHOR environments

This project contains the code for training baseline models on the PointNav task. In this setting the agent
spawns at a location in an environment and is tasked to explore the environment until it finds an object of a
certain type (such as TV or Basketball). Once the agent is confident that it has the object within sight
it executes the `END` action which terminates the episode. If the agent is within a set
distance to the target (in our case 1.5 meters) and the target is visible within its observation frame
the agent succeeded, else it failed.

Provided are experiment configs for training a simple convolutional model with
an GRU using `RGB`, `Depth` or `RGBD` as inputs in
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
python main.py -o storage/objectnav-robothor-rgb -b projects/objectnav_baselines/experiments/robothor/ objectnav_robothor_rgb_resnet_ddppo
```

This trains a simple convolutional neural network with a GRU using RGB input 
passed through a pretrained ResNet-18 visual encoder on the
PointNav task in the RoboTHOR environment and stores the model weights and logs
to `storage/pointnav-robothor-rgb`.
hings you can run with bash commands