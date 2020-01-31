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
`metrics`, and `query_expert` from the base `Task` definition:

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

    def render(self, mode: str = "rgb", *args, **kwargs) -> numpy.ndarray:
        assert mode == "rgb", "only rgb rendering is implemented"
        return self.env.current_frame

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

    def reached_terminal_state(self) -> bool:
        return self._took_end_action

    @classmethod
    def action_names(cls) -> Tuple[str, ...]:
        return cls._actions

    def close(self) -> None:
        self.env.stop()

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

    def metrics(self) -> Dict[str, Any]:
        if not self.is_done():
            return {}
        else:
            return {"success": self._success, "ep_length": self.num_steps_taken()}

    def query_expert(self) -> Tuple[int, bool]:
        return my_objnav_expert_implementation(self)
```

## TaskSampler

