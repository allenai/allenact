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
        scene_period: Optional[int] = None,
        max_tasks: Optional[int] = None,
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

        self.scene_period = scene_period or 0  # default makes a random choice
        self.scene_counter = 0
        self.scene_order = list(range(len(self.scenes)))
        random.shuffle(self.scene_order)
        self.scene_id = 0
        self.max_tasks = max_tasks

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
        return float("inf") if self.max_tasks is None else self.max_tasks

    @property
    def total_unique(self) -> Optional[Union[int, float]]:
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

    def sample_scene(self):
        if self.scene_period == 0:
            # Random scene
            self.scene_id = random.randint(0, len(self.scenes) - 1)
        elif self.scene_counter == self.scene_period:
            if self.scene_id == len(self.scene_order) - 1:
                # Randomize scene order for next iteration
                random.shuffle(self.scene_order)
                # Move to next scene
                self.scene_id = 0
            else:
                # Move to next scene
                self.scene_id += 1
            # Reset scene counter
            self.scene_counter = 1
        else:
            # Stay in current scene
            self.scene_counter += 1

        if self.max_tasks is not None:
            self.max_tasks -= 1

        return self.scenes[self.scene_order[self.scene_id]]

    def next_task(self) -> Optional[ObjectNavTask]:
        if self.max_tasks is not None and len(self) == 0:
            return None

        scene = self.sample_scene()

        if self.env is not None:
            # print("resetting env")
            if scene != self.env.scene_name:
                self.env.reset(scene)
        else:
            # print("creating env")
            self.env = self._create_environment()
            # print("starting env")
            self.env.start(scene_name=scene)

        # print("env up and ready")

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
