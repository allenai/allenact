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

We need to define the methods `action_space`, `render`, `_step`, `reached_terminal_state`, `action_names`, `close`,
`metrics`, and `query_expert` from the base `Task` definition.


### Initialization, action space and termination
Let's start with the definition of the action space and task initialization:
```python
class ObjectNavTask(Task[AI2ThorEnvironment]):
    _actions = (
        'MOVE_AHEAD', 'ROTATE_LEFT', 'ROTATE_RIGHT',
        'LOOK_DOWN', 'LOOK_UP', 'END'
    )

    def __init__(
        self,
        env: AI2ThorEnvironment,
        sensors: List[Sensor],
        task_info: Dict[str, Any],
        max_steps: int,
        **kwargs
    ) -> None:
        super().__init__(
            env=env,
            sensors=sensors,
            task_info=task_info,
            max_steps=max_steps, **kwargs
        )
        self._took_end_action: bool = False
        self._success: Optional[bool] = False

    @property
    def action_space(self):
        return gym.spaces.Discrete(len(self._actions))

    @classmethod
    def action_names(cls) -> Tuple[str, ...]:
        return cls._actions
        def reached_terminal_state(self) -> bool:
        return self._took_end_action

    def close(self) -> None:
        self.env.stop()
    ...
```

### Step method
Next, we define the main method `_step` that will be called every time the agent produces a new action: 
```python
class ObjectNavTask(Task[AI2ThorEnvironment]):
    ...
    def _step(self, action: int) -> RLStepResult:
        action_str = self.action_names()[action]

        if action_str == 'END':
            self._took_end_action = True
            self._success = self._is_goal_object_visible()
            self.last_action_success = self._success
        else:
            self.env.step({"action": action_str})
            self.last_action_success = self.env.last_action_success

        step_result = RLStepResult(
            observation=self.get_observations(),
            reward=self.judge(),
            done=self.is_done(),
            info={"last_action_success": self.last_action_success},
        )
        return step_result

    def _is_goal_object_visible(self) -> bool:
        return any(
            o["objectType"] == self.task_info["object_type"]
            for o in self.env.visible_objects()
        )

    def judge(self) -> float:
        reward = -0.01

        if not self.last_action_success:
            reward += -0.1

        if self._took_end_action:
            reward += 1.0 if self._success else -1.0

        return float(reward)
```

###  Metrics, rendering and expert actions

Finally, we define methods to render and evaluate the current task, and optionally generate expert actions to be used
e.g. for DAGGER training.
```python

    def render(self, mode: str = "rgb", *args, **kwargs) -> numpy.ndarray:
        assert mode == "rgb", "only rgb rendering is implemented"
        return self.env.current_frame


    def metrics(self) -> Dict[str, Any]:
        if not self.is_done():
            return {}
        else:
            return {"success": self._success, "ep_length": self.num_steps_taken()}

    def query_expert(self) -> Tuple[int, bool]:
        return my_objnav_expert_implementation(self)
```

## TaskSampler

We also need to define the corresponding TaskSampler, which must contain implementations for methods `__len__`,
`total_unique`, `last_sampled_task`, `next_task`, `close`, `reset`, and `set_seed`. Currently,
an additional method `all_observation_spaces_equal` is used to ensure compatibility with the current
[RolloutStorage](/api/onpolicy_sync/storage#rolloutstorage).

Let's define a tasks sampler able to provide an infinite number of object navigation tasks for AI2THOR.

### Initialization and termination 

```python
class ObjectNavTaskSampler(TaskSampler):
    def __init__(
        self,
        scenes: List[str],
        object_types: str,
        sensors: List[Sensor],
        max_steps: int,
        env_args: Dict[str, Any],
        action_space: gym.Space,
        seed: Optional[int] = None,
        deterministic_cudnn: bool = False,
        *args,
        **kwargs
    ) -> None:
        self.env_args = env_args
        self.scenes = scenes
        self.object_types = object_types
        self.grid_size = 0.25
        self.env: Optional[AI2ThorEnvironment] = None
        self.sensors = sensors
        self.max_steps = max_steps
        self._action_sapce = action_space

        self.scene_id: Optional[int] = None

        self._last_sampled_task: Optional[ObjectNavTask] = None

        set_seed(seed)

        self.reset()

    def close(self) -> None:
        if self.env is not None:
            self.env.stop()

    def reset(self):
        self.scene_id = 0
    
    def _create_environment(self) -> AI2ThorEnvironment:
        env = AI2ThorEnvironment(
            make_agents_visible=False,
            object_open_speed=0.05,
            restrict_to_initially_reachable_points=True,
            **self.env_args,
        )
        return env
```

# Task sampling

Finally, we need to define methods to determine the number of available tasks (possibly infinite) and sample tasks:
```python

    @property
    def __len__(self) -> Union[int, float]:
        return float("inf")

    @property
    def total_unique(self) -> Optional[Union[int, float]]:
        return None

    @property
    def last_sampled_task(self) -> Optional[ObjectNavTask]:
        return self._last_sampled_task

    @property
    def all_observation_spaces_equal(self) -> bool:
        return True

    def next_task(self) -> Optional[ObjectNavTask]:
        self.scene_id = random.randint(0, len(self.scenes) - 1)
        self.scene = self.scenes[self.scene_id]

        if self.env is not None:
            if scene != self.env.scene_name:
                self.env.reset(scene)
        else:
            self.env = self._create_environment()
            self.env.reset(scene_name=scene)

        self.env.randomize_agent_location()

        object_types_in_scene = set(
            [o["objectType"] for o in self.env.last_event.metadata["objects"]]
        )

        task_info = {"object_type": random.sample(self.object_types, 1)}

        self._last_sampled_task = ObjectNavTask(
            env=self.env,
            sensors=self.sensors,
            task_info=task_info,
            max_steps=self.max_steps,
            action_space=self._action_sapce,
        )
        return self._last_sampled_task
```