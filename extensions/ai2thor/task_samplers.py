import random
import warnings
from typing import List, Dict, Optional, Any, Union

from extensions.ai2thor.environment import AI2ThorEnvironment
from extensions.ai2thor.tasks import ObjectNavTask
from rl_base.sensor import Sensor
from rl_base.task import TaskSampler
import gym


class ObjectNavTaskSampler(TaskSampler):
    def __init__(
        self,
        scenes: List[str],
        object_types: str,
        sensors: List[Sensor],
        max_steps: int,
        env_args: Dict[str, Any],
        action_space: gym.Space,
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

        self._last_sampled_task: Optional[ObjectNavTask] = None

    def _create_environment(self) -> AI2ThorEnvironment:
        env = AI2ThorEnvironment(
            **self.env_args,
            make_agents_visible=False,
            object_open_speed=0.05,
            restrict_to_initially_reachable_points=True,
        )
        return env

    @property
    def __len__(self) -> Union[int, float]:
        """
        @return: Number of total tasks remaining that can be sampled. Can be float('inf').
        """
        return float("inf")

    @property
    def total_unique(self) -> Union[int, float, None]:
        return None

    @property
    def last_sampled_task(self) -> Optional[ObjectNavTask]:
        return self._last_sampled_task

    def close(self) -> None:
        if self.env is not None:
            self.env.stop()

    @property
    def all_observation_spaces_equal(self) -> bool:
        """
        @return: True if all Tasks that can be sampled by this sampler have the
            same observation space. Otherwise False.
        """
        return True

    def next_task(self) -> ObjectNavTask:
        print("creating next task")
        scene = random.choice(self.scenes)

        if self.env is not None:
            print("resetting env")
            self.env.reset(scene)
        else:
            print("creating env")
            self.env = self._create_environment()
            print("starting env")
            self.env.start(scene_name=scene)

        print("env up and ready")

        self.env.randomize_agent_location()

        object_types_in_scene = set(
            [o["objectType"] for o in self.env.last_event.metadata["objects"]]
        )

        task_info = {}
        for ot in random.sample(self.object_types, len(self.object_types)):
            if ot in object_types_in_scene:
                task_info["object_type"] = ot
                break

        if len(task_info) == 0:
            warnings.warn(
                "Scene {} does not contain any"
                " objects of any of the types {}.".format(scene, self.object_types)
            )

        self._last_sampled_task = ObjectNavTask(
            env=self.env,
            sensors=self.sensors,
            task_info=task_info,
            max_steps=self.max_steps,
            action_space=self._action_sapce,
        )
        return self._last_sampled_task
