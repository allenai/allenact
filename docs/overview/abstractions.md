# Primary abstractions

Our package relies on a collection of fundamental abstractions to define a reinforcement learning problem. These
abstractions are described in plain language below. Each of the below sections end with a link to the 
(formal) documentation of the abstraction as well as a link to an example implementation of the abstract (if relevant).

## Task

Tasks define the scope of the interaction between agents and an environment, as well as metrics to evaluate the agents' performance.
 
See the [abstract `Task` class](/api/rl_base/task/#task) 
and an [example implementation](/api/extensions/ai2thor/tasks/#objectnavtask).

## Sensor

Sensors provide observations extracted from an environment or from a task definition (e.g. the end point in point navigation or target object class in semantic navigation) that can be either directly consumed by agents or processed by a [preprocessor](#preprocessor). 

See the [abstract `Sensor` class](/api/rl_base/sensor/#sensor) 
and an [example implementation](/api/extensions/ai2thor/tasks/#objectnavtask).

## Task sampler

A task sampler is responsible for generating a sequence of tasks for agents to solve. The sequence of tasks can be randomly generated (e.g. in training) or extracted from an ordered pool (e.g. in validation or testing).

See the [abstract `TaskSampler` class](/api/rl_base/task/#tasksampler) 
and an [example implementation](/api/extensions/ai2thor/task_samplers/#objectnavtasksampler).

## Actor critic model

The actor-critic agent is responsible for computing batched action probabilities and state values given the observations provided by sensors or preprocessors, internal state representations, previous actions, etc.

See the [abstract `ActorCriticModel` class](/api/onpolicy_sync/policy/#actorcriticmodel) 
and an [example implementation](/api/extensions/ai2thor/task_samplers/#objectnavtasksampler).

## Actor critic loss

Actor-critic losses compute combination of action loss and value loss out of collected experience that can be used to train actor-critic models with back-propagation. 

See the [abstract `AbstractActorCriticLoss` class](/api/onpolicy_sync/policy/#actorcriticmodel) 
and an [example implementation](/api/onpolicy_sync/losses/ppo/#ppo).

## Preprocessor

Preprocessors compute new representations for batched observations from other preprocessors or sensors that can be consumed by either other preprocessors or actor-critic models. 

See the [abstract `Preprocessor` class](/api/rl_base/preprocessor/#preprocessor) 
and an [example implementation](/api/extensions/ai2thor/preprocessors/#resnetpreprocessorthor).

## Rollout storage

Rollout storage is used to store observations, internal states, actions and rewards resulting from the interaction of actor-critic models with tasks running in parallel.

See the [`RolloutStorage` class](/api/onpolicy_sync/storage/#rolloutstorage).
```python
```