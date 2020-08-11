# Tutorial: Swapping in a new environment

## Introduction
This tutorial was designed as a continuation of the `Robothor Pointnav Tutorial` and explains
how to modify the experiment config created in that tutorial to work with the iTHOR and
Habitat environments.

Cross-platform support is one of the key design goals of `embodied-ai`. This is achieved through
a total decoupling of the environment code from the engine, model and algorithm code, so that
swapping in a new environment is as plug and play as possible. Crucially we will be able to 
run a model on different environments without touching the model code at all, which will allow
us to train neural networks in one environment and test them in another.

## RoboTHOR to iTHOR
![iTHOR Framework](../img/iTHOR_framework.png)
Since both the `RoboTHOR` and the `iTHOR` environment stem from the same family and are developed
by the same organization, switching between the two is incredibly easy. We only have to change
the path parameter to point to an iTHOR dataset rather than the RoboTHOR one.

```python
    # Dataset Parameters
    TRAIN_DATASET_DIR = "dataset/ithor/objectnav/train"
    VAL_DATASET_DIR = "dataset/ithor/objectnav/val"
```

That's it!

We might also want to modify the `tag` method to accurately reflect our config but this will not change
the behavior at all and is merely a bookkeeping convenience.
```python
    @classmethod
    def tag(cls):
        return "PointNavRobothorRGBPPO"
```

## RoboTHOR to Habitat
![Habitat Framework](../img/habitat_framework.png)
Since the roboTHOR and Habitat simulators are sufficiently different and have different parameters to configure
this transformation takes a bit more effort, but we only need to modify the environment config and TaskSampler (we
have to change the former because the habitat simulator accepts a different format of configuration and the latter
because the habitat dataset is formatted differently and thus needs to be parsed differently.)

As part of our environment modification, we need to switch from using RoboTHOR sensors to using Habitat sensors.
The implementation of sensors we provide offer an uniform interface across all the environments so we simply have
to swap out our sensor classes:
```python
    SENSORS = [
        DepthSensorHabitat(
            height=SCREEN_SIZE,
            width=SCREEN_SIZE,
            use_resnet_normalization=True,
        ),
        TargetCoordinatesSensorHabitat(coordinate_dims=2),
    ]
```

Next we need to define the simulator config:

```python
    CONFIG = habitat.get_config("configs/gibson.yaml")
    CONFIG.defrost()
    CONFIG.NUM_PROCESSES = NUM_PROCESSES
    CONFIG.SIMULATOR_GPU_IDS = TRAIN_GPUS
    CONFIG.DATASET.SCENES_DIR = "habitat/habitat-api/data/scene_datasets/"
    CONFIG.DATASET.POINTNAVV1.CONTENT_SCENES = ["*"]
    CONFIG.DATASET.DATA_PATH = TRAIN_SCENES
    CONFIG.SIMULATOR.AGENT_0.SENSORS = ["RGB_SENSOR"]
    CONFIG.SIMULATOR.RGB_SENSOR.WIDTH = CAMERA_WIDTH
    CONFIG.SIMULATOR.RGB_SENSOR.HEIGHT = CAMERA_HEIGHT
    CONFIG.SIMULATOR.TURN_ANGLE = 30
    CONFIG.SIMULATOR.FORWARD_STEP_SIZE = 0.25
    CONFIG.ENVIRONMENT.MAX_EPISODE_STEPS = MAX_STEPS

    CONFIG.TASK.TYPE = "Nav-v0"
    CONFIG.TASK.SUCCESS_DISTANCE = 0.2
    CONFIG.TASK.SENSORS = ["POINTGOAL_WITH_GPS_COMPASS_SENSOR"]
    CONFIG.TASK.POINTGOAL_WITH_GPS_COMPASS_SENSOR.GOAL_FORMAT = "POLAR"
    CONFIG.TASK.POINTGOAL_WITH_GPS_COMPASS_SENSOR.DIMENSIONALITY = 2
    CONFIG.TASK.GOAL_SENSOR_UUID = "pointgoal_with_gps_compass"
    CONFIG.TASK.MEASUREMENTS = ["DISTANCE_TO_GOAL", "SPL"]
    CONFIG.TASK.SPL.TYPE = "SPL"
    CONFIG.TASK.SPL.SUCCESS_DISTANCE = 0.2

    CONFIG.MODE = "train"
```
This `CONFIG` object holds very similar values to the ones `ENV_ARGS` held in the RoboTHOR example. We
decided to leave this way of passing in configurations exposed to the user to offer maximum customization
of the underlying environment.

Finally we need to replace the task sampler and its argument generating functions:
```python
    # Define Task Sampler
    from plugins.habitat_plugin.habitat_task_samplers import PointNavTaskSampler
    @classmethod
    def make_sampler_fn(cls, **kwargs) -> TaskSampler:
        return PointNavTaskSampler(**kwargs)

    def train_task_sampler_args(
        self,
        process_ind: int,
        total_processes: int,
        devices: Optional[List[int]] = None,
        seeds: Optional[List[int]] = None,
        deterministic_cudnn: bool = False,
    ) -> Dict[str, Any]:
        config = self.TRAIN_CONFIGS[process_ind]
        return {
            "env_config": config,
            "max_steps": self.MAX_STEPS,
            "sensors": self.SENSORS,
            "action_space": gym.spaces.Discrete(len(PointNavTask.action_names())),
            "distance_to_goal": self.DISTANCE_TO_GOAL,
        }

    def valid_task_sampler_args(
        self,
        process_ind: int,
        total_processes: int,
        devices: Optional[List[int]] = None,
        seeds: Optional[List[int]] = None,
        deterministic_cudnn: bool = False,
    ) -> Dict[str, Any]:
        config = self.CONFIG.clone()
        config.defrost()
        config.DATASET.DATA_PATH = self.VALID_SCENES
        config.MODE = "validate"
        config.freeze()
        return {
            "env_config": config,
            "max_steps": self.MAX_STEPS,
            "sensors": self.SENSORS,
            "action_space": gym.spaces.Discrete(len(PointNavTask.action_names())),
            "distance_to_goal": self.DISTANCE_TO_GOAL,
        }

    def test_task_sampler_args(
        self,
        process_ind: int,
        total_processes: int,
        devices: Optional[List[int]] = None,
        seeds: Optional[List[int]] = None,
        deterministic_cudnn: bool = False,
    ) -> Dict[str, Any]:
        config = self.TEST_CONFIGS[process_ind]
        return {
            "env_config": config,
            "max_steps": self.MAX_STEPS,
            "sensors": self.SENSORS,
            "action_space": gym.spaces.Discrete(len(PointNavTask.action_names())),
            "distance_to_goal": self.DISTANCE_TO_GOAL,
        }
```

As we can see this code looks very similar as well, we simply need to pass slightly different parameters.

## Running a Test
With the setup complete, we should be able to run a test using the exact same command as in the last tutorial:
```bash
python ddmain.py -o projects/pointnav_transfer_turotial/ -c projects/pointnav_robothor_rgb/weights/<REDACTED> -t -b projects/pointnav_robothor_rgb/experiments pointnav_robothor_rgb_ddppo
```
This should test the model trained in RoboTHOR on either iTHOR or Habitat (depending on which modifications we made).

## Conclusion
In this tutorial, we learned how to modify our experiment configurations to work with different environments. By
providing a high level of modularity and out-of-the-box support for both `Habitat` and `THOR`, two of the most popular embodied frameworks out there **embodied-ai** hopes to give researchers the ability to validate their results across many platforms and help guide them towards genuine progress. The source code for this tutorial can be found in `/projects/framework_transfer_tutorial`.
