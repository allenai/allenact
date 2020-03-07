from typing import List, Optional, Union, Dict, Any

from rl_robothor.robothor_environment import RoboThorEnvironment

from rl_base.task import TaskSampler

import random
import warnings
import typing
from typing import Optional, Dict, List, Any
from collections import OrderedDict
import logging

import gym

from rl_robothor.robothor_tasks import ObjectNavTask, PointNavTask
from rl_robothor.robothor_environment import RoboThorEnvironment
# from rl_ai2thor.object_nav.task_samplers import (
#     ObjectNavTaskSampler as BaseObjectNavTaskSampler,
# )
from rl_base.sensor import Sensor
from utils.experiment_utils import set_seed, set_deterministic_cudnn
import copy

LOGGER = logging.getLogger("embodiedrl")


class ObjectNavTaskSampler(TaskSampler):
    def __init__(
        self,
        scenes: List[str],
        object_types: List[str],
        # scene_to_episodes: List[Dict[str, Any]],
        sensors: List[Sensor],
        max_steps: int,
        env_args: Dict[str, Any],
        action_space: gym.Space,
        scene_period: Optional[Union[int, str]] = None,
        max_tasks: Optional[int] = None,
        seed: Optional[int] = None,
        deterministic_cudnn: bool = False,
        fixed_tasks: Optional[List[Dict[str, Any]]] = None,
        allow_flipping: bool = False,
        *args,
        **kwargs
    ) -> None:
        self.env_args = env_args
        self.scenes = scenes
        self.object_types = object_types
        # self.scene_to_episodes = scene_to_episodes
        # self.scene_counters = {scene: -1 for scene in self.scene_to_episodes}
        # self.scenes = list(self.scene_to_episodes.keys())
        self.env: Optional[RoboThorEnvironment] = None
        self.sensors = sensors
        self.max_steps = max_steps
        self._action_space = action_space
        self.allow_flipping = allow_flipping

        self.scene_counter: Optional[int] = None
        self.scene_order: Optional[List[str]] = None
        self.scene_id: Optional[int] = None
        self.scene_period: Optional[
            Union[str, int]
        ] = scene_period  # default makes a random choice
        self.max_tasks: Optional[int] = None
        self.reset_tasks = max_tasks

        self._last_sampled_task: Optional[ObjectNavTask] = None

        self.seed: Optional[int] = None
        self.set_seed(seed)

        if deterministic_cudnn:
            set_deterministic_cudnn()

        self.reset()

    def _create_environment(self) -> RoboThorEnvironment:
        env = RoboThorEnvironment(**self.env_args)
        return env

    @property
    def __len__(self) -> Union[int, float]:
        """Length.

        # Returns

        Number of total tasks remaining that can be sampled. Can be float('inf').
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
        """Check if observation spaces equal.

        # Returns

        True if all Tasks that can be sampled by this sampler have the
            same observation space. Otherwise False.
        """
        return True

    def sample_scene(self, force_advance_scene: bool):
        if force_advance_scene:
            if self.scene_period != "manual":
                LOGGER.warning(
                    "When sampling scene, have `force_advance_scene == True`"
                    "but `self.scene_period` is not equal to 'manual',"
                    "this may cause unexpected behavior."
                )
            self.scene_id = (1 + self.scene_id) % len(self.scenes)
            if self.scene_id == 0:
                random.shuffle(self.scene_order)

        if self.scene_period is None:
            # Random scene
            self.scene_id = random.randint(0, len(self.scenes) - 1)
        elif self.scene_period == "manual":
            pass
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
        elif isinstance(self.scene_period, int):
            # Stay in current scene
            self.scene_counter += 1
        else:
            raise NotImplementedError(
                "Invalid scene_period {}".format(self.scene_period)
            )

        if self.max_tasks is not None:
            self.max_tasks -= 1

        return self.scenes[int(self.scene_order[self.scene_id])]

    # def sample_episode(self, scene):
    #     self.scene_counters[scene] = (self.scene_counters[scene] + 1) % len(self.scene_to_episodes[scene])
    #     if self.scene_counters[scene] == 0:
    #         random.shuffle(self.scene_to_episodes[scene])
    #     return self.scene_to_episodes[scene][self.scene_counters[scene]]

    def next_task(self, force_advance_scene: bool = False) -> Optional[ObjectNavTask]:
        if self.max_tasks is not None and self.max_tasks <= 0:
            return None

        scene = self.sample_scene(force_advance_scene)

        if self.env is not None:
            if scene.replace("_physics", "") != self.env.scene_name.replace(
                "_physics", ""
            ):
                self.env.reset(scene)
        else:
            self.env = self._create_environment()
            self.env.reset(scene_name=scene)

        pose = self.env.randomize_agent_location()

        object_types_in_scene = set(
            [o["objectType"] for o in self.env.last_event.metadata["objects"]]
        )

        task_info = {}
        for ot in random.sample(self.object_types, len(self.object_types)):
            if ot in object_types_in_scene:
                task_info["object_type"] = ot
                break

        if len(task_info) == 0:
            LOGGER.warning(
                "Scene {} does not contain any"
                " objects of any of the types {}.".format(scene, self.object_types)
            )

        # task_info["start_pose"] = OrderedDict(
        #     sorted([(k, float(v)) for k, v in pose.items()], key=lambda x: x[0])
        # )
        task_info['initial_position'] = {k: pose[k] for k in ['x', 'y', 'z']}
        task_info['initial_orientation'] = pose["rotation"]["y"]

        task_info["actions"] = []

        if self.allow_flipping and random.random() > 0.5:
            task_info["mirrored"] = True
        else:
            task_info["mirrored"] = False

        # task_info = copy.deepcopy(self.sample_episode(scene))
        # task_info['actions'] = []
        #
        # pose = {**task_info['initial_position'], 'rotation': {'x': 0.0, 'y': task_info['initial_orientation'], 'z': 0.0}, 'horizon': 0.0}
        # self.env.step({"action": "TeleportFull", **pose})
        # assert self.env.last_action_success, "Failed to initialize agent to {} in {} for epsiode {}".format(pose, scene, task_info)
        # # 'scene': scene,
        # # 'object_type': object_type,
        # # 'object_id': object_id,
        # # 'target_position': target_position,
        # # 'initial_position': pos_unity,
        # # 'initial_orientation': roatation_y,
        # # 'shortest_path': path,
        # # 'shortest_path_length': minimum_path_length

        self._last_sampled_task = ObjectNavTask(
            env=self.env,
            sensors=self.sensors,
            task_info=task_info,
            max_steps=self.max_steps,
            action_space=self._action_space,
        )
        return self._last_sampled_task

    def reset(self):
        self.scene_counter = 0
        self.scene_order = list(range(len(self.scenes)))
        random.shuffle(self.scene_order)
        self.scene_id = 0
        self.max_tasks = self.reset_tasks
        # for scene in self.scene_to_episodes:
        #     random.shuffle(self.scene_to_episodes[scene])
        # for scene in self.scene_counters:
        #     self.scene_counters[scene] = -1

    def set_seed(self, seed: int):
        self.seed = seed
        if seed is not None:
            set_seed(seed)


class PointNavTaskSampler(TaskSampler):
    def __init__(
        self,
        scenes: List[str],
        # object_types: List[str],
        # scene_to_episodes: List[Dict[str, Any]],
        sensors: List[Sensor],
        max_steps: int,
        env_args: Dict[str, Any],
        action_space: gym.Space,
        scene_period: Optional[Union[int, str]] = None,
        max_tasks: Optional[int] = None,
        seed: Optional[int] = None,
        deterministic_cudnn: bool = False,
        fixed_tasks: Optional[List[Dict[str, Any]]] = None,
        *args,
        **kwargs
    ) -> None:
        self.env_args = env_args
        self.scenes = scenes
        # self.object_types = object_types
        # self.scene_to_episodes = scene_to_episodes
        # self.scene_counters = {scene: -1 for scene in self.scene_to_episodes}
        # self.scenes = list(self.scene_to_episodes.keys())
        self.env: Optional[RoboThorEnvironment] = None
        self.sensors = sensors
        self.max_steps = max_steps
        self._action_space = action_space

        self.scene_counter: Optional[int] = None
        self.scene_order: Optional[List[str]] = None
        self.scene_id: Optional[int] = None
        self.scene_period: Optional[
            Union[str, int]
        ] = scene_period  # default makes a random choice
        self.max_tasks: Optional[int] = None
        self.reset_tasks = max_tasks

        self._last_sampled_task: Optional[ObjectNavTask] = None

        self.seed: Optional[int] = None
        self.set_seed(seed)

        if deterministic_cudnn:
            set_deterministic_cudnn()

        self.reset()

    def _create_environment(self) -> RoboThorEnvironment:
        env = RoboThorEnvironment(**self.env_args)
        return env

    @property
    def __len__(self) -> Union[int, float]:
        """Length.

        # Returns

        Number of total tasks remaining that can be sampled. Can be float('inf').
        """
        return float("inf") if self.max_tasks is None else self.max_tasks

    @property
    def total_unique(self) -> Optional[Union[int, float]]:
        # total = 0
        # for scene in self.scene_to_episodes:
        #     total += len(self.scene_to_episodes[scene])
        # return total
        return None

    @property
    def last_sampled_task(self) -> Optional[PointNavTask]:
        return self._last_sampled_task

    def close(self) -> None:
        if self.env is not None:
            self.env.stop()

    @property
    def all_observation_spaces_equal(self) -> bool:
        """Check if observation spaces equal.

        # Returns

        True if all Tasks that can be sampled by this sampler have the
            same observation space. Otherwise False.
        """
        return True

    def sample_scene(self, force_advance_scene: bool):
        if force_advance_scene:
            if self.scene_period != "manual":
                LOGGER.warning(
                    "When sampling scene, have `force_advance_scene == True`"
                    "but `self.scene_period` is not equal to 'manual',"
                    "this may cause unexpected behavior."
                )
            self.scene_id = (1 + self.scene_id) % len(self.scenes)
            if self.scene_id == 0:
                random.shuffle(self.scene_order)

        if self.scene_period is None:
            # Random scene
            self.scene_id = random.randint(0, len(self.scenes) - 1)
        elif self.scene_period == "manual":
            pass
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
        elif isinstance(self.scene_period, int):
            # Stay in current scene
            self.scene_counter += 1
        else:
            raise NotImplementedError(
                "Invalid scene_period {}".format(self.scene_period)
            )

        if self.max_tasks is not None:
            self.max_tasks -= 1

        return self.scenes[int(self.scene_order[self.scene_id])]

    # def sample_episode(self, scene):
    #     self.scene_counters[scene] = (self.scene_counters[scene] + 1) % len(self.scene_to_episodes[scene])
    #     if self.scene_counters[scene] == 0:
    #         random.shuffle(self.scene_to_episodes[scene])
    #     return self.scene_to_episodes[scene][self.scene_counters[scene]]

    def next_task(self, force_advance_scene: bool = False) -> Optional[ObjectNavTask]:
        if self.max_tasks is not None and self.max_tasks <= 0:
            return None

        scene = self.sample_scene(force_advance_scene)

        if self.env is not None:
            if scene.replace("_physics", "") != self.env.scene_name.replace(
                "_physics", ""
            ):
                self.env.reset(scene_name=scene)
        else:
            self.env = self._create_environment()
            self.env.reset(scene_name=scene)

        # task_info = copy.deepcopy(self.sample_episode(scene))
        # task_info['target'] = task_info['target_position']
        # task_info['actions'] = []

        locs = self.env.known_good_locations_list()

        cond = True
        attempt = 0
        while cond and attempt < 10:
            self.env.randomize_agent_location()
            target = copy.copy(random.choice(locs))
            cond = self.env.dist_to_point(target) <= 0
            attempt += 1

        pose = self.env.agent_state()

        task_info = {}
        task_info['scene'] = scene
        task_info['initial_position'] = {k: pose[k] for k in ['x', 'y', 'z']}
        task_info['initial_orientation'] = pose["rotation"]["y"]
        task_info['target'] = target
        task_info['actions'] = []

        if cond:
            LOGGER.warning("No path for sampled episode {}".format(task_info))

        # pose = {**task_info['initial_position'], 'rotation': {'x': 0.0, 'y': task_info['initial_orientation'], 'z': 0.0}, 'horizon': 0.0}
        # self.env.step({"action": "TeleportFull", **pose})
        # assert self.env.last_action_success, "Failed to initialize agent to {} in {} for epsiode {}".format(pose, scene, task_info)

        self._last_sampled_task = PointNavTask(
            env=self.env,
            sensors=self.sensors,
            task_info=task_info,
            max_steps=self.max_steps,
            action_space=self._action_space,
        )
        return self._last_sampled_task

    def reset(self):
        self.scene_counter = 0
        self.scene_order = list(range(len(self.scenes)))
        random.shuffle(self.scene_order)
        self.scene_id = 0
        self.max_tasks = self.reset_tasks

        # for scene in self.scene_to_episodes:
        #     random.shuffle(self.scene_to_episodes[scene])
        # for scene in self.scene_counters:
        #     self.scene_counters[scene] = -1

    def set_seed(self, seed: int):
        self.seed = seed
        if seed is not None:
            set_seed(seed)
