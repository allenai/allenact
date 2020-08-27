## TODO: links to API should be fixed
# Primary abstractions

Our package relies on a collection of fundamental abstractions to define how, and in what task, the model should be trained
and evaluated. A subset of these abstractions are described in plain language below. Each of the below sections end with a link to the 
(formal) documentation of the abstraction as well as a link to an example implementation of the abstract (if relevant).
The following provides a high-level description of how these abstractions interact.


![abstractions-overview](../img/abstractions.png)

## Task

Tasks define the scope of the interaction between agents and an environment (including the action types agents are 
allowed to execute), as well as metrics to evaluate the agents' performance. For example, we might define a task 
`ObjectNavTask` in which agents receive observations obtained from the environment (e.g. RGB images) or directly from 
the task (e.g. a target object class) and are allowed to execute actions such as `MoveAhead`, `RotateRight`, 
`RotateLeft`, and `End` whenever agents determine they have reached their target. The metrics might include a
success indicator or some quantitative metric on the optimality of the followed path.  
 
See the [abstract `Task` class](/api/rl_base/task/#task) 
and an [example implementation](/api/rl_ai2thor/object_nav/tasks/#objectnavtask).

## Sensor

Sensors provide observations extracted from an environment (e.g. RGB or depth images) or directly from a task (e.g. the 
end point in point navigation or target object class in semantic navigation) that can be directly consumed by 
agents.

See the [abstract `Sensor` class](/api/rl_base/sensor/#sensor) 
and an [example implementation](/api/rl_ai2thor/ai2thor_sensors).

## Task sampler

A task sampler is responsible for generating a sequence of tasks for agents to solve. The sequence of tasks can be 
randomly generated (e.g. in training) or extracted from an ordered pool (e.g. in validation or testing).

See the [abstract `TaskSampler` class](/api/rl_base/task/#tasksampler) 
and an [example implementation](/api/rl_ai2thor/object_nav/task_samplers/#objectnavtasksampler).

## Actor critic model

The actor-critic agent is responsible for computing batched action probabilities and state values given the 
observations provided by sensors, internal state representations, previous actions, and potentially 
other inputs.

See the [abstract `ActorCriticModel` class](/api/onpolicy_sync/policy/#actorcriticmodel) 
and an [example implementation](/api/models/object_nav_models/#objectnavtasksampler).

## Actor critic loss

Actor-critic losses compute a combination of action loss and value loss out of collected experience that can be used to 
train actor-critic models with back-propagation, e.g. PPO or A2C.

See the [abstract `AbstractActorCriticLoss` class](/api/onpolicy_sync/losses/abstract_loss#abstractactorcriticloss) 
and an [example implementation](/api/onpolicy_sync/losses/ppo/#ppo).

## Experiment configuration

In `allenact`, experiments are definied by implementing the abstract `ExperimentConfig` class. The methods
of this implementation are then called during training/inference to properly set up the desired experiment. For example,
the `ExperimentConfig.create_model` method will be called at the beginning of training to create the model
to be trained. See either the ["designing your first minigrid experiment"](/tutorials/minigrid-tutorial) or the ["designing an experiment for point navigation"](/tutorials/training-a-pointnav-model)
 tutorials to get an in-depth description of how these experiment configurations are defined in practice.    

See also the [abstract `ExperimentConfig` class](/api/rl_base/experiment_config#experimentconfig) 
and an [example implementation]().