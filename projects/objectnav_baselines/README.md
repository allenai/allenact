# Baseline models ObjectNav (for RoboTHOR/iTHOR)

This project contains the code for training baseline models for the ObjectNav task. In ObjectNav, the agent
spawns at a location in an environment and is tasked to explore the environment until it finds an object of a
certain type (such as TV or Basketball). Once the agent is confident that it has the object within sight
it executes the `END` action which terminates the episode. If the agent is within a set
distance to the target (in our case 1.0 meters) and the target is visible within its observation frame
the agent succeeded, otherwise it failed.

Provided are experiment configs for training a simple convolutional model with
an GRU using `RGB`, `Depth` or `RGB-D` (i.e. `RGB+Depth`) as inputs in
[RoboTHOR](https://ai2thor.allenai.org/robothor/) and [iTHOR](https://ai2thor.allenai.org/ithor/).

The experiments are set up to train models using the [DD-PPO](https://arxiv.org/pdf/1911.00357.pdf)
Reinforcement Learning Algorithm. For the RoboTHOR environment we also have and experiment
(`objectnav_robothor_rgb_resnetgru_dagger.py`) showing how a model can be trained using DAgger,
a form of imitation learning.

To train an experiment run the following command from the `allenact` root directory:

```bash
python main.py <PATH_TO_EXPERIMENT_CONFIG> -o <PATH_TO_OUTPUT> -c
```

Where `<PATH_TO_OUTPUT>` is the path of the directory where we want the model weights
and logs to be stored and `<PATH_TO_EXPERIMENT_CONFIG>` is the path to the python file containing
the experiment configuration. An example usage of this command would be:

```bash
python main.py projects/objectnav_baselines/experiments/robothor/objectnav_robothor_rgb_resnet_ddppo.py -o storage/objectnav-robothor-rgb
```

This trains a simple convolutional neural network with a GRU using RGB input 
passed through a pretrained ResNet-18 visual encoder on the
PointNav task in the RoboTHOR environment and stores the model weights and logs
to `storage/pointnav-robothor-rgb`.

## RoboTHOR ObjectNav 2021 Challenge

The experiment configs found under the `projects/objectnav_baselines/experiments/robothor` directory are designed
to conform to the requirements of the [RoboTHOR ObjectNav 2021 Challenge](https://ai2thor.allenai.org/robothor/cvpr-2021-challenge).
To train a baseline ResNet->GRU model taking RGB-D inputs, run the following command
```bash
python main.py projects/objectnav_baselines/experiments/robothor/objectnav_robothor_rgbd_resnet_ddppo.py -o storage/objectnav-robothor-rgbd
```
Note that, by default, when using a machine with a GPUs, the above experiment will attempt to train using 60 parallel processes
across all available GPUs. See the `TRAIN_GPU_IDS` constant in `experiments/objectnav_thor_base.py` and
the `NUM_PROCESSES` constant in `experiments/robothor/objectnav_robothor_base.py` if you'd like to change which
GPUs are used or how many processes are run respectively.

We provide a pretrained model obtained allowing the above command to run for all 300M training steps and then selecting
the model checkpoint with best validation-set performance (for us occuring at ~170M training steps). You can download 
this model checkpoint by running
```bash
bash pretrained_model_ckpts/download_navigation_model_ckpts.sh robothor-objectnav-challenge-2021
```
from the top-level directory. This will download the pretrained model weights and save them at the path
```bash
pretrained_model_ckpts/robothor-objectnav-challenge-2021/Objectnav-RoboTHOR-RGBD-ResNetGRU-DDPPO/2021-02-09_22-35-15/exp_Objectnav-RoboTHOR-RGBD-ResNetGRU-DDPPO_0.2.0a_300M__stage_00__steps_000170207237.pt
```
You can then run inference on this model (on the test dataset) by running
```bash
export SAVED_MODEL_PATH=pretrained_model_ckpts/robothor-objectnav-challenge-2021/Objectnav-RoboTHOR-RGBD-ResNetGRU-DDPPO/2021-02-09_22-35-15/exp_Objectnav-RoboTHOR-RGBD-ResNetGRU-DDPPO_0.2.0a_300M__stage_00__steps_000170207237.pt
python main.py projects/objectnav_baselines/experiments/robothor/objectnav_robothor_rgbd_resnetgru_ddppo.py -c $SAVED_MODEL_PATH -t 2021-02-09_22-35-15
```
To discourage "cheating", the test dataset has been scrubbed of the information needed to actually compute the success rate / SPL
of your model and so running the above will only save the trajectories your models take. To evaluate these
trajectories you will have to submit them to our leaderboard, see [here for more details](https://github.com/allenai/robothor-challenge/).
If you'd like to get a sense of if your model is doing well before submitting to the leaderboard, you can obtain the 
success rate / SPL of it on our validation dataset. To do this, you can simply comment-out the line
```python
    TEST_DATASET_DIR = os.path.join(os.getcwd(), "datasets/robothor-objectnav/test")
```
within the `projects/objectnav_baselines/experiments/robothor/objectnav_robothor_base.py` file and rerun the above
`python main.py ...` command (when the test dataset is not given, the code defaults to using the validation set).