from math import ceil
from typing import Dict, Any, List, Optional
import random
import copy

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

from allenact.algorithms.onpolicy_sync.losses import PPO
from allenact.algorithms.onpolicy_sync.losses.ppo import PPOConfig
from allenact.base_abstractions.experiment_config import ExperimentConfig, MachineParams
from allenact.base_abstractions.sensor import SensorSuite
from allenact.base_abstractions.task import TaskSampler, BatchedTask
from allenact.utils.experiment_utils import (
    Builder,
    PipelineStage,
    TrainingPipeline,
    LinearDecay,
)
from allenact_plugins.ithor_plugin.ithor_sensors import (
    RGBSensorThor,
    GoalObjectTypeThorSensor,
)
from allenact_plugins.ithor_plugin.ithor_task_samplers import ObjectNavTaskSampler
from allenact_plugins.ithor_plugin.ithor_tasks import ObjectNaviThorGridTask
from allenact_plugins.navigation_plugin.objectnav.models import ObjectNavActorCritic
from allenact_plugins.ithor_plugin.ithor_environment import IThorEnvironment
from allenact.base_abstractions.misc import RLStepResult


class BatchController:
    def __init__(
        self,
        task_batch_size: int,
        **kwargs,
    ):
        self.task_batch_size = task_batch_size
        self.controllers = [IThorEnvironment(**kwargs) for _ in range(max(1, task_batch_size))]
        self._frames = []

    def step(self, actions: List[str]):
        assert len(actions) == self.task_batch_size or len(actions) == self.task_batch_size + 1
        for controller, action in zip(self.controllers, actions):
            controller.step(action=action if action != "End" else "Pass")
        self._frames = []
        return self.batch_last_event()

    def get_agent_location(self):
        return None

    def reset(
        self,
        idx: int,
        scene: str,
    ):
        self.controllers[idx].reset(scene)

    def batch_reset(
        self,
        scenes: List[str],
    ):
        for controller, scene in zip(self.controllers, scenes):
            controller.reset(scene)

    def stop(self):
        for controller in self.controllers:
            controller.stop()

    def last_event(self, idx: int):
        return self.controllers[idx].last_event

    def batch_last_event(self):
        return [controller.last_event for controller in self.controllers]

    def render(self):
        assert len(self._frames) == 0
        for controller in self.controllers:
            self._frames.append(controller.last_event.frame)


class BatchableObjectNaviThorGridTask(ObjectNaviThorGridTask):
    # # TODO BEGIN For compatibility with batch_task_size = 0
    #
    # batch_index = 0
    #
    # def get_observations(self, **kwargs) -> List[Any]:  #-> Dict[str, Any]:
    #     # Render all tasks in batch
    #     self._frames = []
    #     self.env.render()
    #     obs = super().get_observations()
    #     self.env._frames = []
    #     return obs
    #
    # def _step(self, action):
    #     # raise NotImplementedError()
    #     action_str, interm = self._before_env_step(action)
    #     self.env.step([action_str])
    #     self._after_env_step(action, action_str, interm)
    #     return RLStepResult(
    #         observation=self.get_observations(),
    #         reward=self.judge(),
    #         done=self.is_done(),
    #         info={"last_action_success": self.last_action_success},
    #     )
    # # TODO END For compatibility with batch_task_size = 0

    def is_goal_object_visible(self) -> bool:
        """Is the goal object currently visible?"""
        return any(
            o["objectType"] == self.task_info["object_type"]
            for o in self.env.controllers[self.batch_index].visible_objects()
        )

    def _before_env_step(self, action):
        assert isinstance(action, int)

        action_str = self.class_action_names()[action]

        return action_str, None

    def _after_env_step(self, action, action_str, intermediate):
        if action_str == "End":
            self._took_end_action = True
            self._success = self.is_goal_object_visible()
            self.last_action_success = self._success
        else:
            self.last_action_success = self.env.controllers[self.batch_index].last_action_success

            if (
                not self.last_action_success
            ) and self._CACHED_LOCATIONS_FROM_WHICH_OBJECT_IS_VISIBLE is not None:
                self.env.controllers[self.batch_index].update_graph_with_failed_action(failed_action=action_str)

        return RLStepResult(
            observation=None,
            reward=self.judge(),
            done=self.is_done(),
            info={"last_action_success": self.last_action_success},
        )


class BatchedObjectNavTaskSampler(ObjectNavTaskSampler):
    def __init__(self, **kwargs):
        if "task_batch_size" in kwargs:
            self.task_batch_size = kwargs["task_batch_size"]
            self.callback_sensor_suite = kwargs["callback_sensor_suite"]
            kwargs.pop("task_batch_size")
            kwargs.pop("callback_sensor_suite")
        else:
            self.task_batch_size = 0
            self.callback_sensor_suite = None
        super().__init__(**kwargs)

    def _create_environment(self):
        env = BatchController(task_batch_size=self.task_batch_size)
        return env

    def next_task(
        self, force_advance_scene: bool = False, idx=0,
    ) -> Optional[ObjectNaviThorGridTask]:
        if self.max_tasks is not None and self.max_tasks <= 0:
            return None

        scene = self.sample_scene(force_advance_scene)

        if self.env is not None:
            if scene.replace("_physics", "") != self.env.controllers[idx].scene_name.replace(
                "_physics", ""
            ):
                self.env.reset(idx, scene)  # type:ignore
        else:
            self.env = self._create_environment()
            self.env.reset(idx, scene)  # type:ignore

        pose = self.env.controllers[idx].randomize_agent_location()

        object_types_in_scene = set(
            [o["objectType"] for o in self.env.last_event(idx).metadata["objects"]]
        )

        task_info: Dict[str, Any] = {}
        for ot in random.sample(self.object_types, len(self.object_types)):
            if ot in object_types_in_scene:
                task_info["object_type"] = ot
                break

        if len(task_info) == 0:
            print(
                "WARNING",
                "Scene {} does not contain any"
                " objects of any of the types {}.".format(scene, self.object_types)
            )

        task_info["start_pose"] = copy.copy(pose)
        task_info[
            "id"
        ] = f"{scene}__{'_'.join(list(map(str, self.env.controllers[idx].get_key(pose))))}__{task_info['object_type']}"

        if self.task_batch_size > 0:
            self._last_sampled_task = BatchedTask(
                env=self.env,
                sensors=self.sensors,
                task_info=task_info,
                max_steps=self.max_steps,
                action_space=self._action_space,
                task_sampler=self,
                task_classes=[BatchableObjectNaviThorGridTask],
                callback_sensor_suite=self.callback_sensor_suite,
            )
        else:
            self._last_sampled_task = BatchableObjectNaviThorGridTask(
                env=self.env,
                sensors=self.sensors,
                task_info=task_info,
                max_steps=self.max_steps,
                action_space=self._action_space,
            )
        return self._last_sampled_task


class BatchedRGBSensorThor(RGBSensorThor):
    """Sensor for RGB images in THOR.

    Returns from a running IThorEnvironment instance, the current RGB
    frame corresponding to the agent's egocentric view.
    """

    def frame_from_env(
        self, env, task,
    ) -> np.ndarray:  # type:ignore
        assert len(env._frames) > 0
        return env._frames[task.batch_index]


class ObjectNavThorPPOExperimentConfig(ExperimentConfig):
    """A simple object navigation experiment in THOR.

    Training with PPO.
    """

    # A simple setting, train/valid/test are all the same single scene
    # and we're looking for a single object
    OBJECT_TYPES = ["Tomato"]
    TRAIN_SCENES = ["FloorPlan1_physics", "FloorPlan2_physics"]
    # VALID_SCENES = ["FloorPlan1_physics"]
    # TEST_SCENES = ["FloorPlan1_physics"]

    # Setting up sensors and basic environment details
    SCREEN_SIZE = 224
    SENSORS = [
        BatchedRGBSensorThor(
            height=SCREEN_SIZE, width=SCREEN_SIZE, use_resnet_normalization=True,
        ),
        GoalObjectTypeThorSensor(object_types=OBJECT_TYPES),
    ]

    ENV_ARGS = {
        "player_screen_height": SCREEN_SIZE,
        "player_screen_width": SCREEN_SIZE,
        "quality": "Very Low",
    }

    MAX_STEPS = 128
    ADVANCE_SCENE_ROLLOUT_PERIOD: Optional[int] = None
    # VALID_SAMPLES_IN_SCENE = 10
    # TEST_SAMPLES_IN_SCENE = 100

    @classmethod
    def tag(cls):
        return "BatchedObjectNavThorPPO"

    @classmethod
    def training_pipeline(cls, **kwargs):
        ppo_steps = int(1e6)
        lr = 2.5e-4
        num_mini_batch = 2 if not torch.cuda.is_available() else 6
        update_repeats = 4
        num_steps = cls.MAX_STEPS
        metric_accumulate_interval = cls.MAX_STEPS * 1  # Log every 10 max length tasks
        save_interval = 10000
        gamma = 0.99
        use_gae = True
        gae_lambda = 1.0
        max_grad_norm = 0.5

        return TrainingPipeline(
            save_interval=save_interval,
            metric_accumulate_interval=metric_accumulate_interval,
            optimizer_builder=Builder(optim.Adam, dict(lr=lr)),
            num_mini_batch=num_mini_batch,
            update_repeats=update_repeats,
            max_grad_norm=max_grad_norm,
            num_steps=num_steps,
            named_losses={
                "ppo_loss": PPO(clip_decay=LinearDecay(ppo_steps), **PPOConfig),
            },
            gamma=gamma,
            use_gae=use_gae,
            gae_lambda=gae_lambda,
            advance_scene_rollout_period=cls.ADVANCE_SCENE_ROLLOUT_PERIOD,
            pipeline_stages=[
                PipelineStage(loss_names=["ppo_loss"], max_stage_steps=ppo_steps,),
            ],
            lr_scheduler_builder=Builder(
                LambdaLR, {"lr_lambda": LinearDecay(steps=ppo_steps)}
            ),
        )

    @classmethod
    def machine_params(cls, mode="train", **kwargs):
        num_gpus = torch.cuda.device_count()
        has_gpu = num_gpus != 0

        if mode == "train":
            nprocesses = 20 if has_gpu else 2
            gpu_ids = [0] if has_gpu else []
        elif mode == "valid":
            nprocesses = 0
            gpu_ids = [1 % num_gpus] if has_gpu else []
        elif mode == "test":
            nprocesses = 0
            gpu_ids = [0] if has_gpu else []
        else:
            raise NotImplementedError("mode must be 'train', 'valid', or 'test'.")

        return MachineParams(nprocesses=nprocesses, devices=gpu_ids,)

    @classmethod
    def create_model(cls, **kwargs) -> nn.Module:
        return ObjectNavActorCritic(
            action_space=gym.spaces.Discrete(
                len(ObjectNaviThorGridTask.class_action_names())
            ),
            observation_space=SensorSuite(cls.SENSORS).observation_spaces,
            rgb_uuid=cls.SENSORS[0].uuid,
            depth_uuid=None,
            goal_sensor_uuid="goal_object_type_ind",
            hidden_size=512,
            object_type_embedding_dim=8,
        )

    @classmethod
    def make_sampler_fn(cls, **kwargs) -> TaskSampler:
        return BatchedObjectNavTaskSampler(**kwargs)

    @staticmethod
    def _partition_inds(n: int, num_parts: int):
        return np.round(np.linspace(0, n, num_parts + 1, endpoint=True)).astype(
            np.int32
        )

    def _get_sampler_args_for_scene_split(
        self,
        scenes: List[str],
        process_ind: int,
        total_processes: int,
        seeds: Optional[List[int]] = None,
        deterministic_cudnn: bool = False,
    ) -> Dict[str, Any]:
        if total_processes > len(scenes):  # oversample some scenes -> bias
            if total_processes % len(scenes) != 0:
                print(
                    "Warning: oversampling some of the scenes to feed all processes."
                    " You can avoid this by setting a number of workers divisible by the number of scenes"
                )
            scenes = scenes * int(ceil(total_processes / len(scenes)))
            scenes = scenes[: total_processes * (len(scenes) // total_processes)]
        else:
            if len(scenes) % total_processes != 0:
                print(
                    "Warning: oversampling some of the scenes to feed all processes."
                    " You can avoid this by setting a number of workers divisor of the number of scenes"
                )
        inds = self._partition_inds(len(scenes), total_processes)

        return {
            "scenes": scenes[inds[process_ind] : inds[process_ind + 1]],
            "object_types": self.OBJECT_TYPES,
            "env_args": self.ENV_ARGS,
            "max_steps": self.MAX_STEPS,
            "sensors": self.SENSORS,
            "action_space": gym.spaces.Discrete(
                len(ObjectNaviThorGridTask.class_action_names())
            ),
            "seed": seeds[process_ind] if seeds is not None else None,
            "deterministic_cudnn": deterministic_cudnn,
        }

    def train_task_sampler_args(
        self,
        process_ind: int,
        total_processes: int,
        devices: Optional[List[int]] = None,
        seeds: Optional[List[int]] = None,
        deterministic_cudnn: bool = False,
    ) -> Dict[str, Any]:
        res = self._get_sampler_args_for_scene_split(
            self.TRAIN_SCENES,
            process_ind,
            total_processes,
            seeds=seeds,
            deterministic_cudnn=deterministic_cudnn,
        )
        res["scene_period"] = "manual"
        res["env_args"] = {}
        res["env_args"].update(self.ENV_ARGS)
        res["env_args"]["x_display"] = (
            ("0.%d" % devices[process_ind % len(devices)])
            if devices is not None and len(devices) > 0
            else None
        )
        return res
