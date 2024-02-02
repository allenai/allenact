import copy
import gzip
import json
import random
from typing import List, Optional, Union, Dict, Any, cast, Tuple

import gym

from allenact.base_abstractions.sensor import Sensor
from allenact.base_abstractions.task import TaskSampler
from allenact.utils.cache_utils import str_to_pos_for_cache
from allenact.utils.experiment_utils import set_seed, set_deterministic_cudnn
from allenact.utils.system import get_logger
from allenact_plugins.robothor_plugin.robothor_environment import RoboThorEnvironment
from allenact_plugins.robothor_plugin.robothor_tasks import (
    ObjectNavTask,
    PointNavTask,
    NavToPartnerTask,
)


class ObjectNavTaskSampler(TaskSampler):
    def __init__(
        self,
        scenes: Union[List[str], str],
        object_types: List[str],
        sensors: List[Sensor],
        max_steps: int,
        env_args: Dict[str, Any],
        action_space: gym.Space,
        rewards_config: Dict,
        scene_period: Optional[Union[int, str]] = None,
        max_tasks: Optional[int] = None,
        seed: Optional[int] = None,
        deterministic_cudnn: bool = False,
        allow_flipping: bool = False,
        dataset_first: int = -1,
        dataset_last: int = -1,
        **kwargs,
    ) -> None:
        self.rewards_config = rewards_config
        self.env_args = env_args
        self.scenes = scenes
        self.object_types = object_types
        self.env: Optional[RoboThorEnvironment] = None
        self.sensors = sensors
        self.max_steps = max_steps
        self._action_space = action_space
        self.allow_flipping = allow_flipping

        self.scenes_is_dataset = (dataset_first >= 0) or (dataset_last >= 0)

        if not self.scenes_is_dataset:
            assert isinstance(
                self.scenes, List
            ), "When not using a dataset, scenes ({}) must be a list".format(
                self.scenes
            )
            self.scene_counter: Optional[int] = None
            self.scene_order: Optional[List[str]] = None
            self.scene_id: Optional[int] = None
            self.scene_period: Optional[
                Union[str, int]
            ] = scene_period  # default makes a random choice
            self.max_tasks: Optional[int] = None
            self.reset_tasks = max_tasks
        else:
            assert isinstance(
                self.scenes, str
            ), "When using a dataset, scenes ({}) must be a json file name string".format(
                self.scenes
            )
            with open(self.scenes, "r") as f:
                self.dataset_episodes = json.load(f)
                # get_logger().debug("Loaded {} object nav episodes".format(len(self.dataset_episodes)))
            self.dataset_first = dataset_first if dataset_first >= 0 else 0
            self.dataset_last = (
                dataset_last if dataset_last >= 0 else len(self.dataset_episodes) - 1
            )
            assert (
                0 <= self.dataset_first <= self.dataset_last
            ), "dataset_last {} must be >= dataset_first {} >= 0".format(
                dataset_last, dataset_first
            )
            self.reset_tasks = self.dataset_last - self.dataset_first + 1
            # get_logger().debug("{} tasks ({}, {}) in sampler".format(self.reset_tasks, self.dataset_first, self.dataset_last))

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
    def length(self) -> Union[int, float]:
        """Length.

        # Returns

        Number of total tasks remaining that can be sampled. Can be float('inf').
        """
        return float("inf") if self.max_tasks is None else self.max_tasks

    @property
    def total_unique(self) -> Optional[Union[int, float]]:
        return self.reset_tasks

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
                get_logger().warning(
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
        elif self.scene_counter >= cast(int, self.scene_period):
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
            # get_logger().debug("max_tasks {}".format(self.max_tasks))
            return None

        if not self.scenes_is_dataset:
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

            task_info = {"scene": scene}
            for ot in random.sample(self.object_types, len(self.object_types)):
                if ot in object_types_in_scene:
                    task_info["object_type"] = ot
                    break

            if len(task_info) == 0:
                get_logger().warning(
                    "Scene {} does not contain any"
                    " objects of any of the types {}.".format(scene, self.object_types)
                )

            task_info["initial_position"] = {k: pose[k] for k in ["x", "y", "z"]}
            task_info["initial_orientation"] = cast(Dict[str, float], pose["rotation"])[
                "y"
            ]
        else:
            assert self.max_tasks is not None
            next_task_id = self.dataset_first + self.max_tasks - 1
            # get_logger().debug("task {}".format(next_task_id))
            assert (
                self.dataset_first <= next_task_id <= self.dataset_last
            ), "wrong task_id {} for min {} max {}".format(
                next_task_id, self.dataset_first, self.dataset_last
            )
            task_info = copy.deepcopy(self.dataset_episodes[next_task_id])

            scene = task_info["scene"]
            if self.env is not None:
                if scene.replace("_physics", "") != self.env.scene_name.replace(
                    "_physics", ""
                ):
                    self.env.reset(scene_name=scene)
            else:
                self.env = self._create_environment()
                self.env.reset(scene_name=scene)

            self.env.step(
                {
                    "action": "TeleportFull",
                    **{k: float(v) for k, v in task_info["initial_position"].items()},
                    "rotation": {
                        "x": 0.0,
                        "y": float(task_info["initial_orientation"]),
                        "z": 0.0,
                    },
                    "horizon": 0.0,
                    "standing": True,
                }
            )
            assert self.env.last_action_success, "Failed to reset agent for {}".format(
                task_info
            )

            self.max_tasks -= 1

        # task_info["actions"] = []  # TODO populated by Task(Generic[EnvType]).step(...) but unused

        if self.allow_flipping and random.random() > 0.5:
            task_info["mirrored"] = True
        else:
            task_info["mirrored"] = False

        self._last_sampled_task = ObjectNavTask(
            env=self.env,
            sensors=self.sensors,
            task_info=task_info,
            max_steps=self.max_steps,
            action_space=self._action_space,
            reward_configs=self.rewards_config,
        )
        return self._last_sampled_task

    def reset(self):
        if not self.scenes_is_dataset:
            self.scene_counter = 0
            self.scene_order = list(range(len(self.scenes)))
            random.shuffle(self.scene_order)
            self.scene_id = 0
        self.max_tasks = self.reset_tasks

    def set_seed(self, seed: int):
        self.seed = seed
        if seed is not None:
            set_seed(seed)


class ObjectNavDatasetTaskSampler(TaskSampler):
    def __init__(
        self,
        scenes: List[str],
        scene_directory: str,
        sensors: List[Sensor],
        max_steps: int,
        env_args: Dict[str, Any],
        action_space: gym.Space,
        rewards_config: Dict,
        seed: Optional[int] = None,
        deterministic_cudnn: bool = False,
        loop_dataset: bool = True,
        allow_flipping=False,
        env_class=RoboThorEnvironment,
        randomize_materials_in_training: bool = False,
        **kwargs,
    ) -> None:
        self.rewards_config = rewards_config
        self.env_args = env_args
        self.scenes = scenes
        self.episodes = {
            scene: ObjectNavDatasetTaskSampler.load_dataset(
                scene, scene_directory + "/episodes"
            )
            for scene in scenes
        }

        # Only keep episodes containing desired objects
        if "object_types" in kwargs:
            self.episodes = {
                scene: [
                    ep for ep in episodes if ep["object_type"] in kwargs["object_types"]
                ]
                for scene, episodes in self.episodes.items()
            }
            self.episodes = {
                scene: episodes
                for scene, episodes in self.episodes.items()
                if len(episodes) > 0
            }
            self.scenes = [scene for scene in self.scenes if scene in self.episodes]

        self.env_class = env_class
        self.object_types = [
            ep["object_type"] for scene in self.episodes for ep in self.episodes[scene]
        ]
        self.env: Optional[RoboThorEnvironment] = None
        self.sensors = sensors
        self.max_steps = max_steps
        self._action_space = action_space
        self.allow_flipping = allow_flipping
        self.scene_counter: Optional[int] = None
        self.scene_order: Optional[List[str]] = None
        self.scene_id: Optional[int] = None
        # get the total number of tasks assigned to this process
        if loop_dataset:
            self.max_tasks = None
        else:
            self.max_tasks = sum(len(self.episodes[scene]) for scene in self.episodes)
        self.reset_tasks = self.max_tasks
        self.scene_index = 0
        self.episode_index = 0
        self.randomize_materials_in_training = randomize_materials_in_training

        self._last_sampled_task: Optional[ObjectNavTask] = None

        self.seed: Optional[int] = None
        self.set_seed(seed)

        if deterministic_cudnn:
            set_deterministic_cudnn()

        self.reset()

    def _create_environment(self) -> RoboThorEnvironment:
        env = self.env_class(**self.env_args)
        return env

    @staticmethod
    def load_dataset(scene: str, base_directory: str) -> List[Dict]:
        filename = (
            "/".join([base_directory, scene])
            if base_directory[-1] != "/"
            else "".join([base_directory, scene])
        )
        filename += ".json.gz"
        fin = gzip.GzipFile(filename, "r")
        json_bytes = fin.read()
        fin.close()
        json_str = json_bytes.decode("utf-8")
        data = json.loads(json_str)
        random.shuffle(data)
        return data

    @staticmethod
    def load_distance_cache_from_file(scene: str, base_directory: str) -> Dict:
        filename = (
            "/".join([base_directory, scene])
            if base_directory[-1] != "/"
            else "".join([base_directory, scene])
        )
        filename += ".json.gz"
        fin = gzip.GzipFile(filename, "r")
        json_bytes = fin.read()
        fin.close()
        json_str = json_bytes.decode("utf-8")
        data = json.loads(json_str)
        return data

    @property
    def __len__(self) -> Union[int, float]:
        """Length.

        # Returns

        Number of total tasks remaining that can be sampled. Can be float('inf').
        """
        return float("inf") if self.max_tasks is None else self.max_tasks

    @property
    def total_unique(self) -> Optional[Union[int, float]]:
        return self.reset_tasks

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

    @property
    def length(self) -> Union[int, float]:
        """Length.

        # Returns

        Number of total tasks remaining that can be sampled. Can be float('inf').
        """
        return float("inf") if self.max_tasks is None else self.max_tasks

    def next_task(self, force_advance_scene: bool = False) -> Optional[ObjectNavTask]:
        if self.max_tasks is not None and self.max_tasks <= 0:
            return None

        if self.episode_index >= len(self.episodes[self.scenes[self.scene_index]]):
            self.scene_index = (self.scene_index + 1) % len(self.scenes)
            # shuffle the new list of episodes to train on
            random.shuffle(self.episodes[self.scenes[self.scene_index]])
            self.episode_index = 0
        scene = self.scenes[self.scene_index]
        episode = self.episodes[scene][self.episode_index]
        if self.env is None:
            self.env = self._create_environment()

        if scene.replace("_physics", "") != self.env.scene_name.replace("_physics", ""):
            self.env.reset(scene_name=scene)
        else:
            self.env.reset_object_filter()

        self.env.set_object_filter(
            object_ids=[
                o["objectId"]
                for o in self.env.last_event.metadata["objects"]
                if o["objectType"] == episode["object_type"]
            ]
        )

        # only randomize materials in train scenes
        were_materials_randomized = False
        if self.randomize_materials_in_training:
            if (
                "Train" in scene
                or int(scene.replace("FloorPlan", "").replace("_physics", "")) % 100
                < 21
            ):
                were_materials_randomized = True
                self.env.controller.step(action="RandomizeMaterials")

        task_info = {
            "scene": scene,
            "object_type": episode["object_type"],
            "materials_randomized": were_materials_randomized,
        }
        if len(task_info) == 0:
            get_logger().warning(
                "Scene {} does not contain any"
                " objects of any of the types {}.".format(scene, self.object_types)
            )
        task_info["initial_position"] = episode["initial_position"]
        task_info["initial_orientation"] = episode["initial_orientation"]
        task_info["initial_horizon"] = episode.get("initial_horizon", 0)
        task_info["distance_to_target"] = episode.get("shortest_path_length")
        task_info["path_to_target"] = episode.get("shortest_path")
        task_info["object_type"] = episode["object_type"]
        task_info["id"] = episode["id"]
        if self.allow_flipping and random.random() > 0.5:
            task_info["mirrored"] = True
        else:
            task_info["mirrored"] = False

        self.episode_index += 1
        if self.max_tasks is not None:
            self.max_tasks -= 1
        if not self.env.teleport(
            pose=episode["initial_position"],
            rotation=episode["initial_orientation"],
            horizon=episode.get("initial_horizon", 0),
        ):
            return self.next_task()
        self._last_sampled_task = ObjectNavTask(
            env=self.env,
            sensors=self.sensors,
            task_info=task_info,
            max_steps=self.max_steps,
            action_space=self._action_space,
            reward_configs=self.rewards_config,
        )
        return self._last_sampled_task

    def reset(self):
        self.episode_index = 0
        self.scene_index = 0
        self.max_tasks = self.reset_tasks

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
        rewards_config: Dict,
        scene_period: Optional[Union[int, str]] = None,
        max_tasks: Optional[int] = None,
        seed: Optional[int] = None,
        deterministic_cudnn: bool = False,
        **kwargs,
    ) -> None:
        self.rewards_config = rewards_config
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

        self._last_sampled_task: Optional[PointNavTask] = None

        self.seed: Optional[int] = None
        self.set_seed(seed)

        if deterministic_cudnn:
            set_deterministic_cudnn()

        self.reset()

    def _create_environment(self) -> RoboThorEnvironment:
        env = RoboThorEnvironment(**self.env_args)
        return env

    @property
    def length(self) -> Union[int, float]:
        """Length.

        # Returns

        Number of total tasks remaining that can be sampled.
        Can be float('inf').
        """
        return float("inf") if self.max_tasks is None else self.max_tasks

    @property
    def total_unique(self) -> Optional[Union[int, float]]:
        # total = 0
        # for scene in self.scene_to_episodes:
        #     total += len(self.scene_to_episodes[scene])
        # return total
        return self.reset_tasks

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

        True if all Tasks that can be sampled by this sampler
        have the     same observation space. Otherwise False.
        """
        return True

    def sample_scene(self, force_advance_scene: bool):
        if force_advance_scene:
            if self.scene_period != "manual":
                get_logger().warning(
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
        elif self.scene_counter >= cast(int, self.scene_period):
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

    def next_task(self, force_advance_scene: bool = False) -> Optional[PointNavTask]:
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
        # get_logger().debug("locs[0] {} locs[-1] {}".format(locs[0], locs[-1]))

        ys = [loc["y"] for loc in locs]
        miny = min(ys)
        maxy = max(ys)
        assert maxy - miny < 1e-6, "miny {} maxy {} for scene {}".format(
            miny, maxy, scene
        )

        too_close_to_target = True
        target: Optional[Dict[str, float]] = None
        for _ in range(10):
            self.env.randomize_agent_location()
            target = copy.copy(random.choice(locs))
            too_close_to_target = self.env.distance_to_point(target) <= 0
            if not too_close_to_target:
                break

        pose = self.env.agent_state()

        task_info = {
            "scene": scene,
            "initial_position": {k: pose[k] for k in ["x", "y", "z"]},
            "initial_orientation": pose["rotation"]["y"],
            "target": target,
            "actions": [],
        }

        if too_close_to_target:
            get_logger().warning("No path for sampled episode {}".format(task_info))
        # else:
        #     get_logger().debug("Path found for sampled episode {}".format(task_info))

        # pose = {**task_info['initial_position'], 'rotation': {'x': 0.0, 'y': task_info['initial_orientation'], 'z': 0.0}, 'horizon': 0.0}
        # self.env.step({"action": "TeleportFull", **pose})
        # assert self.env.last_action_success, "Failed to initialize agent to {} in {} for epsiode {}".format(pose, scene, task_info)

        self._last_sampled_task = PointNavTask(
            env=self.env,
            sensors=self.sensors,
            task_info=task_info,
            max_steps=self.max_steps,
            action_space=self._action_space,
            reward_configs=self.rewards_config,
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


class PointNavDatasetTaskSampler(TaskSampler):
    def __init__(
        self,
        scenes: List[str],
        scene_directory: str,
        sensors: List[Sensor],
        max_steps: int,
        env_args: Dict[str, Any],
        action_space: gym.Space,
        rewards_config: Dict,
        seed: Optional[int] = None,
        deterministic_cudnn: bool = False,
        loop_dataset: bool = True,
        shuffle_dataset: bool = True,
        allow_flipping=False,
        env_class=RoboThorEnvironment,
        **kwargs,
    ) -> None:
        self.rewards_config = rewards_config
        self.env_args = env_args
        self.scenes = scenes
        self.shuffle_dataset: bool = shuffle_dataset
        self.episodes = {
            scene: ObjectNavDatasetTaskSampler.load_dataset(
                scene, scene_directory + "/episodes"
            )
            for scene in scenes
        }
        self.env_class = env_class
        self.env: Optional[RoboThorEnvironment] = None
        self.sensors = sensors
        self.max_steps = max_steps
        self._action_space = action_space
        self.allow_flipping = allow_flipping
        self.scene_counter: Optional[int] = None
        self.scene_order: Optional[List[str]] = None
        self.scene_id: Optional[int] = None
        # get the total number of tasks assigned to this process
        if loop_dataset:
            self.max_tasks = None
        else:
            self.max_tasks = sum(len(self.episodes[scene]) for scene in self.episodes)
        self.reset_tasks = self.max_tasks
        self.scene_index = 0
        self.episode_index = 0

        self._last_sampled_task: Optional[PointNavTask] = None

        self.seed: Optional[int] = None
        self.set_seed(seed)

        if deterministic_cudnn:
            set_deterministic_cudnn()

        self.reset()

    def _create_environment(self) -> RoboThorEnvironment:
        env = self.env_class(**self.env_args)
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
        return self.reset_tasks

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

    def next_task(self, force_advance_scene: bool = False) -> Optional[PointNavTask]:
        if self.max_tasks is not None and self.max_tasks <= 0:
            return None

        if self.episode_index >= len(self.episodes[self.scenes[self.scene_index]]):
            self.scene_index = (self.scene_index + 1) % len(self.scenes)
            # shuffle the new list of episodes to train on
            if self.shuffle_dataset:
                random.shuffle(self.episodes[self.scenes[self.scene_index]])
            self.episode_index = 0

        scene = self.scenes[self.scene_index]
        episode = self.episodes[scene][self.episode_index]
        if self.env is not None:
            if scene.replace("_physics", "") != self.env.scene_name.replace(
                "_physics", ""
            ):
                self.env.reset(scene_name=scene, filtered_objects=[])
        else:
            self.env = self._create_environment()
            self.env.reset(scene_name=scene, filtered_objects=[])

        def to_pos(s):
            if isinstance(s, (Dict, Tuple)):
                return s
            if isinstance(s, float):
                return {"x": 0, "y": s, "z": 0}
            return str_to_pos_for_cache(s)

        for k in ["initial_position", "initial_orientation", "target_position"]:
            episode[k] = to_pos(episode[k])

        task_info = {
            "scene": scene,
            "initial_position": episode["initial_position"],
            "initial_orientation": episode["initial_orientation"],
            "target": episode["target_position"],
            "shortest_path": episode["shortest_path"],
            "distance_to_target": episode["shortest_path_length"],
            "id": episode["id"],
        }

        if self.allow_flipping and random.random() > 0.5:
            task_info["mirrored"] = True
        else:
            task_info["mirrored"] = False

        self.episode_index += 1
        if self.max_tasks is not None:
            self.max_tasks -= 1

        if not self.env.teleport(
            pose=episode["initial_position"], rotation=episode["initial_orientation"]
        ):
            return self.next_task()

        self._last_sampled_task = PointNavTask(
            env=self.env,
            sensors=self.sensors,
            task_info=task_info,
            max_steps=self.max_steps,
            action_space=self._action_space,
            reward_configs=self.rewards_config,
        )

        return self._last_sampled_task

    def reset(self):
        self.episode_index = 0
        self.scene_index = 0
        self.max_tasks = self.reset_tasks

    def set_seed(self, seed: int):
        self.seed = seed
        if seed is not None:
            set_seed(seed)

    @property
    def length(self) -> Union[int, float]:
        """Length.

        # Returns

        Number of total tasks remaining that can be sampled.
        Can be float('inf').
        """
        return float("inf") if self.max_tasks is None else self.max_tasks


class NavToPartnerTaskSampler(TaskSampler):
    def __init__(
        self,
        scenes: List[str],
        sensors: List[Sensor],
        max_steps: int,
        env_args: Dict[str, Any],
        action_space: gym.Space,
        rewards_config: Dict,
        scene_period: Optional[Union[int, str]] = None,
        max_tasks: Optional[int] = None,
        seed: Optional[int] = None,
        deterministic_cudnn: bool = False,
        **kwargs,
    ) -> None:
        self.rewards_config = rewards_config
        self.env_args = env_args
        self.scenes = scenes
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

        self._last_sampled_task: Optional[NavToPartnerTask] = None

        self.seed: Optional[int] = None
        self.set_seed(seed)

        if deterministic_cudnn:
            set_deterministic_cudnn()

        self.reset()

    def _create_environment(self) -> RoboThorEnvironment:
        assert (
            self.env_args["agentCount"] == 2
        ), "NavToPartner is only defined for 2 agents!"
        env = RoboThorEnvironment(**self.env_args)
        return env

    @property
    def length(self) -> Union[int, float]:
        """Length.

        # Returns

        Number of total tasks remaining that can be sampled.
        Can be float('inf').
        """
        return float("inf") if self.max_tasks is None else self.max_tasks

    @property
    def total_unique(self) -> Optional[Union[int, float]]:
        return self.reset_tasks

    @property
    def last_sampled_task(self) -> Optional[NavToPartnerTask]:
        return self._last_sampled_task

    def close(self) -> None:
        if self.env is not None:
            self.env.stop()

    @property
    def all_observation_spaces_equal(self) -> bool:
        """Check if observation spaces equal.

        # Returns

        True if all Tasks that can be sampled by this sampler
        have the     same observation space. Otherwise False.
        """
        return True

    def sample_scene(self, force_advance_scene: bool):
        if force_advance_scene:
            if self.scene_period != "manual":
                get_logger().warning(
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
        elif self.scene_counter >= cast(int, self.scene_period):
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

    def next_task(
        self, force_advance_scene: bool = False
    ) -> Optional[NavToPartnerTask]:
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

        too_close_to_target = True
        for _ in range(10):
            self.env.randomize_agent_location(agent_id=0)
            self.env.randomize_agent_location(agent_id=1)

            pose1 = self.env.agent_state(0)
            pose2 = self.env.agent_state(1)

            def retry_dist(position: Dict[str, float], object_type: Dict[str, float]):
                allowed_error = 0.05
                debug_log = ""
                d = -1.0
                while allowed_error < 2.5:
                    d = self.env.distance_from_point_to_point(
                        position, object_type, allowed_error
                    )
                    if d < 0:
                        debug_log = (
                            f"In scene {self.env.scene_name}, could not find a path from {position} to {object_type} with"
                            f" {allowed_error} error tolerance. Increasing this tolerance to"
                            f" {2 * allowed_error} any trying again."
                        )
                        allowed_error *= 2
                    else:
                        break
                if d < 0:
                    get_logger().debug(
                        f"In scene {self.env.scene_name}, could not find a path from {position} to {object_type}"
                        f" with {allowed_error} error tolerance. Returning a distance of -1."
                    )
                elif debug_log != "":
                    get_logger().debug(debug_log)
                return d

            dist = self.env.distance_cache.find_distance(
                self.env.scene_name,
                {k: pose1[k] for k in ["x", "y", "z"]},
                {k: pose2[k] for k in ["x", "y", "z"]},
                retry_dist,
            )

            too_close_to_target = (
                dist <= 1.25 * self.rewards_config["max_success_distance"]
            )
            if not too_close_to_target:
                break

        task_info = {
            "scene": scene,
            "initial_position1": {k: pose1[k] for k in ["x", "y", "z"]},
            "initial_position2": {k: pose2[k] for k in ["x", "y", "z"]},
            "initial_orientation1": pose1["rotation"]["y"],
            "initial_orientation2": pose2["rotation"]["y"],
            "id": "_".join(
                [scene]
                # + ["%4.2f" % pose1[k] for k in ["x", "y", "z"]]
                # + ["%4.2f" % pose1["rotation"]["y"]]
                # + ["%4.2f" % pose2[k] for k in ["x", "y", "z"]]
                # + ["%4.2f" % pose2["rotation"]["y"]]
                + ["%d" % random.randint(0, 2 ** 63 - 1)]
            ),
        }

        if too_close_to_target:
            get_logger().warning("Bad sampled episode {}".format(task_info))

        self._last_sampled_task = NavToPartnerTask(
            env=self.env,
            sensors=self.sensors,
            task_info=task_info,
            max_steps=self.max_steps,
            action_space=self._action_space,
            reward_configs=self.rewards_config,
        )
        return self._last_sampled_task

    def reset(self):
        self.scene_counter = 0
        self.scene_order = list(range(len(self.scenes)))
        random.shuffle(self.scene_order)
        self.scene_id = 0
        self.max_tasks = self.reset_tasks

    def set_seed(self, seed: int):
        self.seed = seed
        if seed is not None:
            set_seed(seed)
