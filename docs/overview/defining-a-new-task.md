# Defining a new task

In order to use new tasks in our experiments, we need to define two classes:

* A [Task](/api/rl_base/task#task), including, among others, a `step` implementation providing a
[RLStepResult](/api/rl_base/common#rlstepresult), a `metrics` method providing quantitative performance measurements 
for agents and, optionally, a `query_expert` method that can be used e.g. with an
[imitation loss](/api/onpolicy_sync/losses/imitation#imitation) during training.
* A [TaskSampler](/api/rl_base/task#tasksampler), that allows instantiating new Tasks for the agents to solve during
training, validation and testing.

## Task

Let's define a semantic navigation task, where agents have to navigate from a starting point in an environment to an
object of a specific class using a minimal amount of steps and deciding when the goal has been reached.

```python
```

## TaskSampler