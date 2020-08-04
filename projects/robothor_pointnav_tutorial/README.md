# RoboTHOR PointNav Tutorial

## Introduction
One of the most obvious tasks that an embodied agent should master is navigating the world it inhabits.
Before we can teach a robot to cook or clean it first needs to be able to move around. The simplest
way to formulate "moving around" into a task is by making your agent find a beacon somewhere in the environment.
This beacon transmits its location, such that at any time, the agent can get the direction and euclidian distance
to the beacon. This particular task is often called Point Navigation, or **PointNav** for short.

#### Pointnav
At first glance, this task seems trivial. If the agent is given the direction and distance of the target at
all times, can it not simply follow this signal directly? The answer is no, because agents are often trained
on this task in environments that emulate real-world buildings which are not wide-open spaces, but rather
contain many smaller rooms. Because of this, the agent has to learn to navigate human spaces and use doors
and hallways to efficiently navigate from one side of the house to the other. This task becomes particularly
difficult when the agent is tested in an environment that it is not trained in. If the agent does not know
how the floor plan of an environment looks, it has to learn to predict the design of man-made structures,
to efficiently navigate across them, much like how people instinctively know how to move around a building
they have never seen before based on their experience navigating similar buildings.

#### What is an environment anyways?
Environments are worlds in which embodied agents exist. If our embodied agent is simply a neural network that is being trained in a simulator, then that simulator is its environment. Similarly, if our agent is a
physical robot then its environment is the real world. The agent interacts with the environment by taking one
of several available actions (such as "move forward", or "turn left"). After each action, the environment
produces a new frame that the agent can analyze to determine its next step. For many tasks, including PointNav
the agent also has a special "stop" action which indicates that the agent thinks it has reached the target.
After this action is called the agent will be reset to a new location, regardless if it reached the
target. The hope is that after enough training the agent will learn to correctly assess that it has successfully
navigated to the target.

There are many simulators designed for the training
of embodied agents. In this tutorial, we will be using a simulator called RoboTHOR, 
which is designed specifically to train models that can easily be transferred to a real robot, by providing a
photo-realistic virtual environment and a real-world replica of the environment that researchers can have access to. 
RoboTHOR contains 60 different virtual scenes with different floor plans and furniture and 15 validation scenes.

It is also important to mention that **embodied-rl**
has a class abstraction called Environment. This is not the actual simulator game engine or robotics controller,
but rather a shallow wrapper that provides a uniform interface to the actual environment.

#### Learning algorithm
Finally, let us briefly touch on the algorithm that we will use to train our embodied agent to navigate. While
*embodied-rl* offers us great flexibility to train models using complex pipelines, we will be using a simple
pure reinforcement learning approach for this tutorial. More specifically, we will be using DD-PPO,
a decentralized and distributed variant of the ubiquitous PPO algorithm. For those unfamiliar with Reinforcement
Learning we highly recommend this tutorial by Andrej Karpathy (http://karpathy.github.io/2016/05/31/rl/), and this book by Sutton and Barto (http://www.incompleteideas.net/book/the-book-2nd.html). Essentially what we are doing
is letting our agent explore the environment on its own, rewarding it for taking actions that bring it closer
to its goal and punishing it for actions that take it away from its goal. We then optimize the agent's model
to maximize this reward.


## Dataset Setup
To train the model on the PointNav task, we need to download the dataset and precomputed cache of distances to the target. The dataset contains a list of episodes with thousands of randomly generated starting positions and target locations for each of the scenes. The precomputed cache of distances is a large dictionary containing
the shortest path from each point in a scene, to every other point in that scene. This is used to reward the agent
for moving closer to the target in terms of geodesic distance - the actual path distance (as opposed to a 
straight line distance).

We can download and unzip the data with the following commands:
```wget <REDACTED>```
```unzip <REDACTED>```


## Config File Setup
Now comes the most important part of the tutorial, we are going to write an experiment config file.

Unlike a library that can be imported into python, **embodied-rl** is structured as a framework with a runner script called `ddmain.py` which will run the experiment specified in a config file. This design forces us to keep meticulous records of exactly which settings were used to produce a particular result,
which can be very useful given how expensive RL models are to train.

We will start by creating a new directory inside the `projects` directory. We can name this whatever we want but for now, we will go with `robothor_pointnav_tutuorial`. Then we can create a directory called 
 `experiments` inside the new directory we just created. This hierarchy is not necessary but it helps keep
our experiments neatly organized. Now we create a file called `pointnav_robothor_rgb_ddppo` inside the
`experiments` folder (again the name of this file is arbitrary).

We start off by importing `ExperimentConfig` from the framework and defining a new subclass:
```
from rl_base.experiment_config import ExperimentConfig
class ObjectNavRoboThorRGBPPOExperimentConfig(ExperimentConfig):
```
We then define the task parameters. For PointNav, these include the maximum number of steps our agent
can take before being reset (this prevents the agent from wondering on forever), and a configuration
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
In this case, we set the maximum number of steps to 500.
We give the agent a reward of -0.01 for each action that it takes (this is to encourage it to reach the goal
in as few actions as possible), and a reward of 10.0 if the agent manages to successfully reach its destination.
If the agent selects the `stop` action without reaching the target we do not punish it (although this is
sometimes useful for preventing the agent from stopping prematurely). Finally, our agent gets rewarded if it moves
closer to the target and gets punished if it moves further. `shaping_weight` controls how strong this signal should
be and is here set to 1.0. These parameters work well for training an agent on PointNav, but feel free to play around
with them.

Next, we set the parameters of the simulator itself. Here we select a resolution at which the engine will render
every frame (640 by 480) and a resolution at which the image will be fed into the neural network (here it is set
to a 224 by 224 box).
```
    # Simulator Parameters
    CAMERA_WIDTH = 640
    CAMERA_HEIGHT = 480
    SCREEN_SIZE = 224
```

Next, we set the hardware parameters for the training engine. `NUM_PROCESSES` sets the total number of parallel processes that will be used to train the model. In general, more processes result in faster training, but since each process is a unique instance of the environment in which we are training they can take up a lot of memory. Depending on the size of the model, the environment, and the hardware we are using, we may need to adjust this number, but for a setup with 8 GTX Titans, 60 processes work fine. 60 also happens to be the number of training scenes in RoboTHOR, which allows each process to load only a single scene into memory, saving time and space.
 
 
 `TRAINING_GPUS` takes the ids of the GPUS on which
the model should be trained. Similarly `VALIDATION_GPUS` and `TESTING_GPUS` hold the ids of the GPUS on which the validation and testing will occur. During training, a validation process is constantly running and evaluating
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

Since we are using a dataset to train our model we need to define the path to where we have stored it. If we
download the dataset instructed above we can define the path as follows
```
    # Dataset Parameters
    TRAIN_DATASET_DIR = "dataset/robothor/objectnav/train"
    VAL_DATASET_DIR = "dataset/robothor/objectnav/val"
```


Next, we define the sensors. `RGBSensorThor` is the environment's implementation of an RGB sensor. It takes the
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

For the sake of this example, we are also going to be using a preprocessor with our model. In *embodied-rl*
the preprocessor abstraction is designed with large models with frozen weights in mind. These models often
hail from the ResNet family and transform the raw pixels that our agent observes in the environment, into a
complex embedding, which then gets stored and used as input to our trainable model instead of the original image.
Most other preprocessing work is done in the sensor classes (as we just saw with the RGB
sensor scaling and normalizing our input), but for the sake of efficiency, all neural network preprocessing should
use this abstraction.
```
    PREPROCESSORS = [
            Builder(ResnetPreProcessorHabitat,
                {
                    "input_height": SCREEN_SIZE,
                    "input_width": SCREEN_SIZE,
                    "output_width": 7,
                    "output_height": 7,
                    "output_dims": 512,
                    "pool": False,
                    "torchvision_resnet_model": models.resnet18,
                    "input_uuids": ["rgb_lowres"],
                    "output_uuid": "rgb_resnet",
                    "parallel": False,  # TODO False for debugging
                }
            ),
    ]
```

Next, we must define all of the observation inputs that our model will use. These are just
the hardcoded ids of the sensors we are using in the experiment.
```    
    OBSERVATIONS = [
        "rgb_resnet",
        "target_coordinates_ind",
    ]
```

Finally, we must define the settings of our simulator. We set the camera dimensions to the values
we defined earlier. We set rotateStepDegrees to 30 degrees, which means that every time the agent takes a
turn action, they will rotate by 30 degrees. We set grid size to 0.25 which means that every time the
agent moves forward, it will do so by 0.25 meters. 
```    
    ENV_ARGS = dict(
            width=CAMERA_WIDTH,
            height=CAMERA_HEIGHT,
            rotateStepDegrees=30.0,
            gridSize=0.25,
    )
```


Now we move on to the methods that we must define to finish implementing an experiment config. Firstly we
have a simple method that just returns the name of the experiment.

```
   @classmethod
    def tag(cls):
        return "PointNavRobothorRGBPPO"
```



Next, we define the training pipeline. In this function, we specify exactly which algorithm or algorithms
we will use to train our model. In this simple example, we are using the PPO loss with a learning rate of 3e-4.
We specify 250 million steps of training and a rollout length of 30 with the `ppo_steps` and `num_steps` parameters
respectively. All the other standard PPO parameters are also present in this function. `metric_accumulate_interval`
sets the frequency at which data is accumulated from all the processes and logged while `save_interval` sets how
often we save the model weights and run validation on them.

```
    @classmethod
    def training_pipeline(cls, **kwargs):
        ppo_steps = int(250000000)
        lr = 3e-4
        num_mini_batch = 1
        update_repeats = 3
        num_steps = 30
        save_interval = 5000000
        metric_accumulate_interval = 10000
        gamma = 0.99
        use_gae = True
        gae_lambda = 0.95
        max_grad_norm = 0.5
        return TrainingPipeline(
            save_interval=save_interval,
            metric_accumulate_interval=metric_accumulate_interval,
            optimizer_builder=Builder(optim.Adam, dict(lr=lr)),
            num_mini_batch=num_mini_batch,
            update_repeats=update_repeats,
            max_grad_norm=max_grad_norm,
            num_steps=num_steps,
            named_losses={"ppo_loss": Builder(PPO, kwargs={}, default=PPOConfig, )},
            gamma=gamma,
            use_gae=use_gae,
            gae_lambda=gae_lambda,
            advance_scene_rollout_period=cls.ADVANCE_SCENE_ROLLOUT_PERIOD,
            pipeline_stages=[
                PipelineStage(loss_names=["ppo_loss"], max_stage_steps=ppo_steps)
            ],
            lr_scheduler_builder=Builder(
                LambdaLR, {"lr_lambda": LinearDecay(steps=ppo_steps)}
            ),
        )
```


We define the helper method `split_num_processes` to split the different scenes that we want to train with
amongst the different available devices. "machine_params" returns the hardware parameters of each
process, based on the list of devices we defined above.
```
    def split_num_processes(self, ndevices):
        assert self.NUM_PROCESSES >= ndevices, "NUM_PROCESSES {} < ndevices".format(self.NUM_PROCESSES, ndevices)
        res = [0] * ndevices
        for it in range(self.NUM_PROCESSES):
            res[it % ndevices] += 1
        return res

    def machine_params(self, mode="train", **kwargs):
        if mode == "train":
            workers_per_device = 1
            gpu_ids = [] if not torch.cuda.is_available() else self.TRAINING_GPUS * workers_per_device
            nprocesses = 1 if not torch.cuda.is_available() else self.split_num_processes(len(gpu_ids))
            sampler_devices = self.TRAINING_GPUS
            render_video = False
        elif mode == "valid":
            nprocesses = 1
            gpu_ids = [] if not torch.cuda.is_available() else self.VALIDATION_GPUS
            render_video = False
        elif mode == "test":
            nprocesses = 1
            gpu_ids = [] if not torch.cuda.is_available() else self.TESTING_GPUS
            render_video = False
        else:
            raise NotImplementedError("mode must be 'train', 'valid', or 'test'.")

        # Disable parallelization for validation process
        if mode == "valid":
            for prep in self.PREPROCESSORS:
                prep.kwargs["parallel"] = False

        observation_set = Builder(ObservationSet, kwargs=dict(
            source_ids=self.OBSERVATIONS, all_preprocessors=self.PREPROCESSORS, all_sensors=self.SENSORS
        )) if mode == 'train' or nprocesses > 0 else None

        return {
            "nprocesses": nprocesses,
            "gpu_ids": gpu_ids,
            "sampler_devices": sampler_devices if mode == "train" else gpu_ids,
            "observation_set": observation_set,
            "render_video": render_video,
        }
```

Now we define the actual model that we will be using. **embodied-rl** offers first-class support for PyTorch,
so any PyTorch model will work here, as long as its forward method accepts a dictionary with sensor names as
keys and their input tensors as values. Here we borrow a model from the `pointnav_baselines` project (which
unsurprisingly contains several PointNav baselines). It is a small convolutional network that expects the output of a ResNet as its rgb input followed by a single-layered GRU. The model accepts as input the number of different
actions our agent can perform in the environment through the `action_space` parameter, which we get from the task definition. We also define the shape of the inputs we are going to be passing to the model with `observation_space`
We specify the names of our sensors with `goal_sensor_uuid` and `rgb_resnet_preprocessor_uuid`. Finally, we define
the size of our RNN with `hidden_layer` and the size of the embedding of our goal sensor data (the direction and
distance to the target) with `goal_dims`.
```
    # Define Model
    @classmethod
    def create_model(cls, **kwargs) -> nn.Module:
        return ResnetTensorPointNavActorCritic(
            action_space=gym.spaces.Discrete(len(PointNavTask._actions)),
            observation_space=kwargs["observation_set"].observation_spaces,
            goal_sensor_uuid="target_coordinates_ind",
            rgb_resnet_preprocessor_uuid="rgb_resnet",
            hidden_size=512,
            goal_dims=32,
        )
```

We also need to define the task sampler that we will be using. This is a piece of code that generates instances
of tasks for our agent to perform (essentially starting locations and targets for PointNav). Since we are getting
our tasks from a dataset, the task sampler is a very simple code that just reads the specified file and sets
the agent to the next starting locations whenever the agent exceeds the maximum number of steps or selects the
`stop` action.
```
    # Define Task Sampler
    @classmethod
    def make_sampler_fn(cls, **kwargs) -> TaskSampler:
        return PointNavDatasetTaskSampler(**kwargs)
```
You might notice that we did not specify the task sampler's arguments, but are rather passing them in. The
reason for this is that each process will have its own task sampler, and we need to specify exactly which scenes
each process should work with. If we have several GPUS and many scenes this process of distributing the work can be rather complicated so we define a few helper functions to do just this.
```
    # Utility Functions for distributing scenes between GPUs
    @staticmethod
    def _partition_inds(n: int, num_parts: int):
        return np.round(np.linspace(0, n, num_parts + 1, endpoint=True)).astype(
            np.int32
        )

    def _get_sampler_args_for_scene_split(
        self,
        scenes_dir: str,
        process_ind: int,
        total_processes: int,
        seeds: Optional[List[int]] = None,
        deterministic_cudnn: bool = False,
    ) -> Dict[str, Any]:
        path = scenes_dir + "*.json.gz" if scenes_dir[-1] == "/" else scenes_dir + "/*.json.gz"
        scenes = [scene.split("/")[-1].split(".")[0] for scene in glob.glob(path)]
        if total_processes > len(scenes):  # oversample some scenes -> bias
            if total_processes % len(scenes) != 0:
                print(
                    "Warning: oversampling some of the scenes to feed all processes."
                    " You can avoid this by setting a number of workers divisible by the number of scenes"
                )
            scenes = scenes * int(ceil(total_processes / len(scenes)))
            scenes = scenes[: total_processes * (len(scenes) // total_processes)]
        else:
            if len(scenes) % total_processes != 0:
                print(
                    "Warning: oversampling some of the scenes to feed all processes."
                    " You can avoid this by setting a number of workers divisor of the number of scenes"
                )
        inds = self._partition_inds(len(scenes), total_processes)

        return {
            "scenes": scenes[inds[process_ind]:inds[process_ind + 1]],
            "max_steps": self.MAX_STEPS,
            "sensors": self.SENSORS,
            "action_space": gym.spaces.Discrete(len(PointNavTask._actions)),
            "seed": seeds[process_ind] if seeds is not None else None,
            "deterministic_cudnn": deterministic_cudnn,
            "rewards_config": self.REWARD_CONFIG
        }
```

The very last things we need to define are the sampler arguments themselves. We define them separately for a train,
validation, and test sampler, but in this case, they are almost the same. The arguments need to include the location
of the dataset and distance cache as well as the environment arguments for our simulator, both of which we defined above
and are just referencing here. The only consequential differences between these task samplers are the path to the dataset we are using (train or validation) and whether we want to loop over the dataset or not (we want this for training since we want to train for several epochs, but we do not need this for validation and testing). Since the test scenes of RoboTHOR are private we are also testing on our validation
set.
```
    def train_task_sampler_args(
        self,
        process_ind: int,
        total_processes: int,
        devices: Optional[List[int]] = None,
        seeds: Optional[List[int]] = None,
        deterministic_cudnn: bool = False,
    ) -> Dict[str, Any]:
        res = self._get_sampler_args_for_scene_split(
            self.TRAIN_DATASET_DIR + '/episodes/',
            process_ind,
            total_processes,
            seeds=seeds,
            deterministic_cudnn=deterministic_cudnn,
        )
        res["scene_directory"] = self.TRAIN_DATASET_DIR
        res["loop_dataset"] = True
        res["env_args"] = {}
        res["env_args"].update(self.ENV_ARGS)
        res["env_args"]["x_display"] = (
            ("0.%d" % devices[process_ind % len(devices)]) if devices is not None and len(devices) > 0 else None
        )
        res["allow_flipping"] = True
        return res

    def valid_task_sampler_args(
        self,
        process_ind: int,
        total_processes: int,
        devices: Optional[List[int]] = None,
        seeds: Optional[List[int]] = None,
        deterministic_cudnn: bool = False,
    ) -> Dict[str, Any]:
        res = self._get_sampler_args_for_scene_split(
            self.VAL_DATASET_DIR + '/episodes/',
            process_ind,
            total_processes,
            seeds=seeds,
            deterministic_cudnn=deterministic_cudnn,
        )
        res["scene_directory"] = self.VAL_DATASET_DIR
        res["loop_dataset"] = False
        res["env_args"] = {}
        res["env_args"].update(self.ENV_ARGS)
        res["env_args"]["x_display"] = (
            ("0.%d" % devices[process_ind % len(devices)]) if devices is not None and len(devices) > 0 else None
        )
        return res

    def test_task_sampler_args(
            self,
            process_ind: int,
            total_processes: int,
            devices: Optional[List[int]] = None,
            seeds: Optional[List[int]] = None,
            deterministic_cudnn: bool = False,
    ) -> Dict[str, Any]:
        res = self._get_sampler_args_for_scene_split(
            self.VAL_DATASET_DIR + '/episodes/',
            process_ind,
            total_processes,
            seeds=seeds,
            deterministic_cudnn=deterministic_cudnn,
        )
        res["scene_directory"] = self.VAL_DATASET_DIR
        res["loop_dataset"] = False
        res["env_args"] = {}
        res["env_args"].update(self.ENV_ARGS)
        res["env_args"]["x_display"] = "10.0"
        return res

```

This is it! If we copy all of the code into a file we should be able to run our experiment!


## Testing Pre-Trained Model
With the experiment all set up, we can try testing it with pre-trained weights.

We can download and unzip these weights with the following commands:
```
mkdir projects/pointnav_robothor_rgb/weights
cd projects/pointnav_robothor_rgb/weights
cd models
wget <REDACTED>
unzip magic
```
We can then test the model by running:
```
python ddmain.py -o <PATH_TO_OUTPUT> -c <PATH_TO_CHECKPOINT> -t -b <BASE_DIRECTORY_OF_YOUR_EXPERIMENT> <EXPERIMENT_NAME>
```
Where `PATH_TO_OUTPUT` is the location where the results of the test will be dumped, `PATH_TO_CHECKPOINT` is the 
location of the downloaded model weights, `<BASE_DIRECTORY_OF_YOUR_EXPERIMENT>` is a path to the directory where our
experiment definition is stored and <EXPERIMENT_NAME> is simply the name of our experiment (without the file extension).

For our current setup the following command would work:
```
python ddmain.py -o projects/pointnav_robothor_rgb/storage/ -c projects/pointnav_robothor_rgb/weights/NAME -t -b projects/pointnav_robothor_rgb/experiments pointnav_robothor_rgb_ddppo
```
The scripts should produce a json output in the specified folder containing the results of our test.

## Training Model From Scratch
We can also train the model from scratch by running:
```
python ddmain.py -o <PATH_TO_OUTPUT> -c -b <BASE_DIRECTORY_OF_YOUR_EXPERIMENT> <EXPERIMENT_NAME>
```
But be aware, training this takes nearly 2 days on a machine with 8 GPU. For our current setup the following command would work:
```
python ddmain.py -o projects/pointnav_robothor_rgb/storage/ -b projects/pointnav_robothor_rgb/experiments pointnav_robothor_rgb_ddppo
```
If we start up a tensorboard server during training and specify that `output_dir=storage` the output should look
something like this:
**insert photo**


## Conclusion
In this tutorial, we learned how to create a new PointNav experiment using **embodied-rl**. There are many simple
and obvious ways to modify the experiment from here - changing the model, the learning algorithm and the environment
each requires very few lines of code changed in the above file, allowing us to explore our embodied ai research ideas
across different frameworks with ease.
