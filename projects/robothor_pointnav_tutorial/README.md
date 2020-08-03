# RoboTHOR PointNav Tutorial

## Introduction
One of the most obvious tasks that an embodied agent should master is navigating the world it inhabits.
Before we can teach a robot to cook, or clean it first needs to bea able to move around. The simplest
way to formulate "moving around" into a task is by making your agent find a beacon somewhere in the environment.
This beacon transmits its location, such that at any time, the agent can get the direction and euclidian distance
to the beacon. This particular task is often called Point Navigation, or **PointNav** for short.

#### Pointnav
At first glance this tasks seems trivial. If the agent is given the direction and distance of the target at
all time, can it not simply follow this signal directly? The answer is no, because agents are often trained
on this tasks in environments that emulate real world buildings which are not wide open spaces, but rather
contain many smaller rooms. Because of this, the agent has to learn to navigate human spaces and use doors
and hallways to efficiently navigate from one side of the house to the other. This task becomes particularly
difficult when the agent is tested in an environment that it is not trained in. If the agent does not know
how the floor plan of an environment looks, it has to learn to predict the design of man made structures,
to efficiently navigate across them, much like how people instinctively know how to move around a building
they have never seen before based on their experience navigating similar buildings.

#### What is an environment anyways?
Environments are worlds in which embodied agents exist. If our embodied agent is simply a neural network 
that is being trained in a simulator, than that simulator is its environment. Similarly if our agent is a
physical robot then its environment is the real world. There are many simulators designed for the training
of embodied agents. In this tutorial we will be using a simulator called RoboTHOR, 
that is designed specifically to be tested on a real robot, in a controlled real world environment that looks
visually very similar to the simulated environment. RoboTHOR contains 60 different virtual scenes with different
 floor plans and furniture, and 15 validation scenes.

It is also important to mention that **embodied-rl**
has a class abstraction called Environment. This is not the actual simulator game engine or robotics controller,
but rather a shallow wrapper that provides a uniform interface.

#### Learning algorithm
Finally let us briefly touch on the algorithm that we will use to train our embodied agent to navigate.

## Dataset Setup
To train the model on the pointnav task, we need to download the dataset and precomputed cache of distances 
to the target. The dataset contains a list of episodes with thousands of randomly generated starting positions
and target locations for each of the scenes. The precomputed cache of distances is a large dictionary containing
the shortest path from each point in a scene, to every other point in that scene. This is used to reward the agent
for moving closer to the target in terms of geodesic distance - the actual path distance (as opposed to a 
straight line distance).

We can download and unzip the data with the following commands:
```wget magic_link```
```unzip macuv_link```


## Config File Setup
Now comes the most important part of the tutorial, we are going to write an experiment config file.

Unlike a library that can be imported into python, **embodied-rl** is structured as a framework with 
a runner script called `main.py` which will run the experiment specified in a config file. This design 
forces us to keep meticulous records of exactly which settings were used to produce a particular result,
which can be very useful given how expensive RL models are to train.

We will start by creating a new directory inside the `projects` directory. We can name this whatever we want
 but for now we will go with `robothor_pointnav_tutuorial`. Then we can create a directory called 
 `experiments` inside the new directory we just created. This hierarchy is not necessary but it helps keep
our experiments neatly organized. Now we create a file called `pointnav_robothor_rgb_ddppo` inside the
`experiments` folder (again the name of this file is arbitrary).

We then start off by importing `ExperimentConfig` from the framework and defining a new subclass:
```
from rl_base.experiment_config import ExperimentConfig
class ObjectNavRoboThorRGBPPOExperimentConfig(ExperimentConfig):
```
We then define the task parameters. For pointnav these include the maximum number of steps our agent
can take before being reset (this prevents the agrent from wondering on forever), and a configuration
for the reward function that we will be using. 

```
    # Task Parameters
    MAX_STEPS = 500
    REWARD_CONFIG = {
        "step_penalty": -0.01,
        "goal_success_reward": 10.0,
        "failed_stop_reward": 0.0,
        "shaping_weight": 1.0,
    }
```
In this case we set the maximum number of steps to 500.
We give the agent a reward of -0.01 for each action that it takes (this is to encourage it to reach the goal
in as few actions as possible), and a reward of 10.0 if the agent manages to successfully reach its destination.
If the agent selects the `stop` action without reaching the target we do not punish it (although this is
sometimes useful for preventing the agent from stopping prematurely). Finally our agent gets rewarded if it moves
closer to the target and gets punished if it moves further. `shaping_weight` controls how strong this signal should
be and is here set to 1.0. These parameters work well for training an agent on pointnav, but feel free to play around
with them.

Next we set the parameters of the simulator itself. Here we select a resolution at which the engine will render
every frame (640 by 480) and a resolution at which the image will be fed into the neural network (here it is set
to a 224 by 224 box).
```
    # Simulator Parameters
    CAMERA_WIDTH = 640
    CAMERA_HEIGHT = 480
    SCREEN_SIZE = 224
```

Next we set the hardware parameters for the training engine. `NUM_PROCESSES` sets the total number of
 parallel processes that will be used to train the model. In general more processes result in faster
 training, but since each process is a unique instance of the environment in which we are training they can take 
 up a lot of memory. Depending on the size of the model, the environment and the hardware we are using
 we may need to adjust this number, but for a setup with 8 GTX Titans, 60 processes work fine. 60 also happens to
 be the number of training scenes in RoboTHOR, which allows each process to load only as single scene into memory,
 saving time and space.
 
 
 `TRAINING_GPUS` takes the ids of the GPUS on which
the model should be trained. Similarly `VALIDATION_GPUS` and `TESTING_GPUS` hold the ids of the GPUS on which 
the validation and testing will occur. During training, a validation process is constantly running and evaluating
the current model, to show the progress on the validation set, so reserving a GPU for validation can be a good idea.
If our hardware setup does not include a GPU, these fields can be set to empty lists, as the codebase will default
to running everything on the CPU with only 1 process.
```
    # Training Engine Parameters
    ADVANCE_SCENE_ROLLOUT_PERIOD = 10000000000000
    NUM_PROCESSES = 60
    TRAINING_GPUS = [0, 1, 2, 3, 4, 5, 6]
    VALIDATION_GPUS = [7]
    TESTING_GPUS = [7]
```

Next we define the sensors. `RGBSensorThor` is the environment's implementation of an RGB sensor. It takes the
raw image outputted by the simulator and resizes it, to the input dimensions for our neural network that we
specified above. It also performs normalization if we want. `GPSCompassSensorRoboThor` is a sensor that tracks
the point our agent needs to move to. It tells us the direction and distance to our goal at every time step.
```
SENSORS = [
        RGBSensorThor(
            {
                "height": SCREEN_SIZE,
                "width": SCREEN_SIZE,
                "use_resnet_normalization": True,
                "uuid": "rgb_lowres",
            }
        ),
        GPSCompassSensorRoboThor({}),
    ]

```



## Testing Pre-Trained Model
With the model all set up, we can try testing it with pre trained weights.

We can download and unzip these weights with the following commands:
```
mkdir projects/pointnav_robothor_rgb/weights
cd projects/pointnav_robothor_rgb/weights
cd models
wget magic
unzip magic
```
We can then test the model by running:
```
python ddmain.py -o <PATH_TO_OUTPUT> -c <PATH_TO_CHECKPOINT> -t -b <BASE_DIRECTORY_OF_YOUR_EXPERIMENT> <EXPERIMENT_NAME>
```
Where `PATH_TO_OUTPUT` is the location where the results of the test will be dumped, `PATH_TO_CHECKPOINT` is the 
location of the downloaded model weights, `<BASE_DIRECTORY_OF_YOUR_EXPERIMENT>` is a oath to the directory where our
experiment definition is stored and <EXPERIMENT_NAME> is simply the name of our experiment (without the file extension).

For our current setup the following command would work:
```
python ddmain.py -o projects/pointnav_robothor_rgb/storage/ -c projects/pointnav_robothor_rgb/weights/NAME -t -b projects/pointnav_robothor_rgb/experiments pointnav_robothor_rgb_ddppo
```
The scripts should produce results that look like this:
**PHOTO**

## Training Model From Scratch