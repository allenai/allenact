# OpenAI gym for continuous control from AllenAct

In this tutorial, we:
1. Introduce the `gym_plugin`, which enables some of the tasks in [OpenAI's gym](https://gym.openai.com/) for training
and inference within AllenAct.
1. Show an example of continuous control with an arbitrary action space covering 2 policies for one of the `gym` tasks.


## The task

For this tutorial, we'll focus on one of the continuous-control environments under the `Box2D` group of `gym`
environments: [LunarLanderContinuous-v2](https://gym.openai.com/envs/LunarLanderContinuous-v2/). In this task, the goal
is to smoothly land a lunar module in a landing pad. 

![The LunarLanderContinuous-v2 task](../img/lunar_lander_continuous_demo.png).

To achieve this goal, we need to provide continuous control for a
main engine and directional one (2 real values). In order to solve the task, the expected reward is of at least 200
points. The controls for main and directional engines are both in the range -1.0 to +1.0 and the observation space is
composed of 8 scalars indicating `x` and `y` positions, `x` and `y` velocities, lander angle and angular velocity, and left and 
right ground contact. Note that these 8 scalars provide a full observation of the state.


## Implementation

For this tutorial, we'll use the readily available `gym_plugin`, which includes a
[wrapper for `gym` environments](../api/plugins/gym_plugin/gym_environment.md#gymenvironment), a
[task sampler](../api/plugins/gym_plugin/gym_tasks.md#gymtasksampler) and
[task definition](../api/plugins/gym_plugin/gym_tasks.md#gymcontinuousbox2dtask), a
[sensor](../api/plugins/gym_plugin/gym_sensors.md#gymbox2dsensor) to wrap the observations provided by the `gym`
environment, and a simple [model](../api/plugins/gym_plugin/gym_models.md#memorylessactorcritic).

The experiment config, similar to the one used for the
[Navigation in MiniGrid tutorial](tutorials/minigrid-tutorial.md), is defined as follows:

### Sensors and Model

As mentioned above, we'll use a [GymBox2DSensor](../api/plugins/gym_plugin/gym_sensors.md#gymbox2dsensor) to provide
full observations from the state of the `gym` environment to our model.

We define our `ActorCriticModel` agent using a lightweight implementation with separate MLPs for actors and critic,
[MemorylessActorCritic](../api/plugins/gym_plugin/gym_models.md#memorylessactorcritic). Since
this is a model for continuous control, note that the superclass of our model is `ActorCriticModel[GaussianDistr]`
instead of `ActorCriticModel[CategoricalDistr]`, since we'll use a
[Gaussian distribution](../api/plugins/gym_plugin/gym_distributions.md#gaussiandistr) to sample actions.

### Task samplers

We use an available `TaskSampler` implementation for `gym` environments that allows to sample
[GymTasks](../api/plugins/gym_plugin/gym_tasks.md#gymtask):
[GymTaskSampler](../api/plugins/gym_plugin/gym_tasks.md#gymtasksampler). Even though it is possible to let the task
sampler instantiate the proper sensor for the chosen task name (by passing `None`), we use the sensors we instantiate
above, which contain a custom identifier (`gym_box_data`) also used by the model.

Similarly to what we do in the Minigrid navigation tutorial, the task sampler will sample random tasks for ever, while,
during testing (or validation), we sample a fixed number of tasks (as otherwise we would never finish evaluating).
In `allenact` this is made possible by defining different arguments for the task sampler in each mode:

Note that we just sample 3 tasks for validation and testing in this case, which suffice to illustrate the model success.

### Machine parameters

Given the simplicity of the task and model, we can just train the model on the CPU. During training, success should
reach 100% in less than 10 minutes, whereas solving the task might (reward > 200) take a bit more than a half hour
(on a laptop CPU).

We allocate a larger number of samplers for training (8) than for validation or testing (just 1), and we default to CPU
usage by returning an empty list of `devices`.

### Training pipeline

The last definition required before starting to train is a training pipeline. In this case, we use two PPO
stages (one with strong entropy constraints to favor sampling and one where we allow the model to become deterministic)
with linearly decaying learning rate:


## Training and validation

We have a complete implementation of this experiment's configuration class in `projects/tutorials/gym_tutorial.py`.
To start training from scratch, we just need to invoke

```bash
python main.py gym_tutorial -b projects/tutorials -m 8 -o /PATH/TO/gym_output -s 12345 -e
```

from the `allenact` root directory. Note that we include `-e` to enforce deterministic evaluation. Please refer to the
[Navigation in MiniGrid tutorial](tutorials/minigrid-tutorial.md) if in doubt of the meaning of the rest of parameters.

If we have Tensorboard installed, we can track progress with
```bash
tensorboard --logdir /PATH/TO/gym_output
```
which will default to the URL [http://localhost:6006/](http://localhost:6006/).

After 2,500,000 steps, the script will terminate. The training curves should look similar to:

![training curves](../img/gym_train.png)

If everything went well, the `valid` success rate should quickly converge to 1 and the mean reward to around 250.
The validation curves should look similar to:

![validation curves](../img/gym_valid.png)


## Testing

The training start date for the experiment, in `YYYY-MM-DD_HH-MM-SS` format, is used as the name of one of the
subfolders in the path to the checkpoints, saved under the output folder.
In order to test for a specific experiment, we need to pass its training start date with the option
`-t EXPERIMENT_DATE`:

```bash
python main.py minigrid_tutorial -b projects/tutorials -m 1 -o /PATH/TO/gym_output -s 12345 -e -t EXPERIMENT_DATE
```

Again, if everything went well, the `test` success rate should converge to 1 and the mean reward to around 250.
The test curves should look similar to:

![test curves](../img/minigrid_test.png)
"""
