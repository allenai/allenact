# Primary abstractions

Our package relies on a collection of fundamental abstractions to define a reinforcement learning problem. These
abstractions are described in plain language below. Each of the below sections end with a link to the 
(formal) documentation of the abstraction as well as a link to an example implementation of the abstract (if relevant).

## Task
 
See the [abstract `Task` class](/api/rl_base/task/#task) 
and an [example implementation](/api/extensions/ai2thor/tasks/#objectnavtask).

## Sensor

See the [abstract `Sensor` class](/api/rl_base/sensor/#sensor) 
and an [example implementation](/api/extensions/ai2thor/tasks/#objectnavtask).

## Task sampler

See the [abstract `TaskSampler` class](/api/rl_base/task/#tasksampler) 
and an [example implementation](/api/extensions/ai2thor/task_samplers/#objectnavtasksampler).

## Actor critic model

See the [abstract `ActorCriticModel` class](/api/onpolicy_sync/policy/#actorcriticmodel) 
and an [example implementation](/api/extensions/ai2thor/task_samplers/#objectnavtasksampler).

## Actor critic loss

See the [abstract `AbstractActorCriticLoss` class](/api/onpolicy_sync/policy/#actorcriticmodel) 
and an [example implementation](/api/onpolicy_sync/losses/ppo/#ppo).

## Preprocessor

See the [abstract `Preprocessor` class](/api/rl_base/preprocessor/#preprocessor) 
and an [example implementation](/api/extensions/ai2thor/preprocessors/#resnetpreprocessorthor).

## Rollout storage

See the [`RolloutStorage` class](/api/onpolicy_sync/storage/#rolloutstorage).
```python
```