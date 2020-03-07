from typing import Dict, Any, List, Optional
import json
from math import ceil

import gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torchvision import models
import numpy as np

from onpolicy_sync.losses.a2cacktr import A2CConfig
from models.frcnn_tensor_object_nav_models import ResnetFasterRCNNTensorsObjectNavActorCritic
from onpolicy_sync.losses import A2C
from rl_base.experiment_config import ExperimentConfig
from rl_base.task import TaskSampler
from rl_base.preprocessor import ObservationSet
from rl_robothor.robothor_tasks import ObjectNavTask
from rl_robothor.robothor_task_samplers import ObjectNavTaskSampler
from rl_ai2thor.ai2thor_sensors import RGBSensorThor, GoalObjectTypeThorSensor
from rl_habitat.habitat_preprocessors import ResnetPreProcessorHabitat
from rl_robothor.robothor_preprocessors import FasterRCNNPreProcessorRoboThor
from utils.experiment_utils import Builder, PipelineStage, TrainingPipeline, LinearDecay


class ObjectNavRoboThorBaseExperimentConfig(ExperimentConfig):
    """An Object Navigation experiment configuration in RoboThor"""

    TRAIN_SCENES = [
        "FloorPlan_Train%d_%d" % (wall, furniture)
        for wall in range(1, 13)
        for furniture in range(1, 6)
    ]

    VALID_SCENES = [
        "FloorPlan_Val%d_%d" % (wall, furniture)
        for wall in range(1, 4)
        for furniture in range(1, 6)
    ]

    TEST_SCENES = [
        "FloorPlan_test-dev%d_%d" % (wall, furniture)
        for wall in range(1, 3)
        for furniture in range(1, 3)
    ]

    CAMERA_WIDTH = 400  # 640
    CAMERA_HEIGHT = 300  # 480

    SCREEN_SIZE = 224

    DETECTOR_DETS = 3

    MAX_STEPS = 200
    ADVANCE_SCENE_ROLLOUT_PERIOD = 10  # if more than 1 scene per worker

    VALIDATION_SAMPLES_PER_SCENE = 1

    NUM_PROCESSES = 60  # TODO 2 for debugging

    TARGET_TYPES = sorted(
        [
            'AlarmClock',
            'Apple',
            'BasketBall',
            'Mug',
            'Television',
        ]
    )

    TARGET_TO_DETECTOR_MAP = {
        'AlarmClock': 'clock',
        'Apple': 'apple',
        'BasketBall': 'sports ball',
        'Mug': 'cup',
        'Television': 'tv',
    }

    DETECTOR_TYPES = FasterRCNNPreProcessorRoboThor.COCO_INSTANCE_CATEGORY_NAMES

    SENSORS = [
        RGBSensorThor(
            {
                "height": SCREEN_SIZE,
                "width": SCREEN_SIZE,
                "use_resnet_normalization": True,
                "uuid": "rgb_lowres",
            }
        ),
        RGBSensorThor(
            {
                "height": CAMERA_HEIGHT,
                "width": CAMERA_WIDTH,
                "use_resnet_normalization": False,
                "uuid": "rgb_highres"
            }
        ),
        GoalObjectTypeThorSensor({
            "object_types": TARGET_TYPES,
            "target_to_detector_map": TARGET_TO_DETECTOR_MAP,
            "detector_types": DETECTOR_TYPES,
        }),
    ]

    PREPROCESSORS = [
        ResnetPreProcessorHabitat(
            config={
                "input_height": SCREEN_SIZE,
                "input_width": SCREEN_SIZE,
                "output_width": 7,
                "output_height": 7,
                "output_dims": 512,
                "pool": False,
                "torchvision_resnet_model": models.resnet18,
                "input_uuids": ["rgb_lowres"],
                "output_uuid": "rgb_resnet",
                "parallel": False,
            }
        ),
        FasterRCNNPreProcessorRoboThor(
            config={
                "input_height": CAMERA_HEIGHT,
                "input_width": CAMERA_WIDTH,
                "max_dets": DETECTOR_DETS,
                "detector_spatial_res": 7,
                "detector_thres": 0.12,
                "input_uuids": ["rgb_highres"],
                "output_uuid": "object_detector",
                "parallel": True,  # TODO False for debugging
            }
        )
    ]

    OBSERVATIONS = [
        "rgb_resnet",
        "goal_object_type_ind",
        "object_detector"
    ]

    ENV_ARGS = dict(
        width=CAMERA_WIDTH,
        height=CAMERA_HEIGHT,
        continuousMode=True,
        applyActionNoise=True,
        agentType="stochastic",
        rotateStepDegrees=45.0,
        visibilityDistance=1.5,
        gridSize=0.25,
        snapToGrid=False,
        agentMode="bot",
    )

    @classmethod
    def tag(cls):
        return "ObjectNav"

    @classmethod
    def training_pipeline(cls, **kwargs):
        a2c_steps = int(1e8)
        lr = 1e-3
        num_mini_batch = 1
        update_repeats = 4
        num_steps = 30
        save_interval = 200000
        log_interval = 2000
        gamma = 0.99
        use_gae = True
        gae_lambda = 0.95
        max_grad_norm = 0.5
        return TrainingPipeline(
            save_interval=save_interval,
            log_interval=log_interval,
            optimizer_builder=Builder(optim.Adam, dict(lr=lr)),
            num_mini_batch=num_mini_batch,
            update_repeats=update_repeats,
            max_grad_norm=max_grad_norm,
            num_steps=num_steps,
            named_losses={"a2c_loss": Builder(A2C, kwargs={}, default=A2CConfig,)},
            gamma=gamma,
            use_gae=use_gae,
            gae_lambda=gae_lambda,
            advance_scene_rollout_period=cls.ADVANCE_SCENE_ROLLOUT_PERIOD,
            pipeline_stages=[
                PipelineStage(loss_names=["a2c_loss"], end_criterion=a2c_steps)
            ],
            # lr_scheduler_builder=Builder(
            #     LambdaLR, {"lr_lambda": LinearDecay(steps=a2c_steps)}
            # ),
        )

    def machine_params(self, mode="train", **kwargs):
        if mode == "train":
            nprocesses = 1 if not torch.cuda.is_available() else self.NUM_PROCESSES  # TODO default 2 for debugging
            sampler_devices = [0, 1, 2, 3, 4, 5, 6, 7]
            gpu_ids = [] if not torch.cuda.is_available() else [0]
            render_video = False
        elif mode == "valid":
            nprocesses = 1  # TODO debugging (0)
            if not torch.cuda.is_available():
                gpu_ids = []
            else:
                gpu_ids = [0]
            render_video = False
        elif mode == "test":
            nprocesses = 1
            if not torch.cuda.is_available():
                gpu_ids = []
            else:
                gpu_ids = [0]
            render_video = True
        else:
            raise NotImplementedError("mode must be 'train', 'valid', or 'test'.")

        observation_set = ObservationSet(
            self.OBSERVATIONS, self.PREPROCESSORS, self.SENSORS
        )

        return {
            "nprocesses": nprocesses,
            "gpu_ids": gpu_ids,
            "sampler_devices": sampler_devices if mode == "train" else gpu_ids,
            "observation_set": observation_set,
            "render_video": render_video,
        }

    @classmethod
    def create_model(cls, **kwargs) -> nn.Module:
        return ResnetFasterRCNNTensorsObjectNavActorCritic(
            action_space=gym.spaces.Discrete(len(ObjectNavTask.action_names())),
            observation_space=kwargs["observation_set"].observation_spaces,
            goal_sensor_uuid="goal_object_type_ind",
            resnet_preprocessor_uuid="rgb_resnet",
            detector_preprocessor_uuid="object_detector",
            rnn_hidden_size=512,
            goal_dims=32,
            max_dets=3,
        )

    @classmethod
    def make_sampler_fn(cls, **kwargs) -> TaskSampler:
        return ObjectNavTaskSampler(**kwargs)

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
            "scenes": scenes[inds[process_ind]:inds[process_ind + 1]],
            "object_types": self.TARGET_TYPES,
            "max_steps": self.MAX_STEPS,
            "sensors": self.SENSORS,
            "action_space": gym.spaces.Discrete(len(ObjectNavTask.action_names())),
            "seed": seeds[process_ind] if seeds is not None else None,
            "deterministic_cudnn": deterministic_cudnn,
            "rewards_config": {
                "step_penalty": -0.01,
                "goal_success_reward": 5.0,
                "unsuccessful_action_penalty": -0.05,
                "failed_stop_reward": -1.0,
                "shaping_weight": 0.0,  # applied to the decrease in distance to target
                "exploration_shaping_weight": 0.1,  # relative to shaping weight
            },
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
            ("0.%d" % devices[process_ind % len(devices)]) if devices is not None and len(devices) > 0 else None
        )
        res["allow_flipping"] = True
        return res

    def valid_task_sampler_args(
        self,
        process_ind: int,
        total_processes: int,
        devices: Optional[List[int]] = None,
        seeds: Optional[List[int]] = None,
        deterministic_cudnn: bool = False,
    ) -> Dict[str, Any]:
        res = self._get_sampler_args_for_scene_split(
            self.VALID_SCENES,
            process_ind,
            total_processes,
            seeds=seeds,
            deterministic_cudnn=deterministic_cudnn,
        )
        res["scene_period"] = self.VALIDATION_SAMPLES_PER_SCENE
        res["max_tasks"] = self.VALIDATION_SAMPLES_PER_SCENE * len(res["scenes"])
        res["env_args"] = {}
        res["env_args"].update(self.ENV_ARGS)
        res["env_args"]["x_display"] = (
            ("0.%d" % devices[process_ind % len(devices)]) if devices is not None and len(devices) > 0 else None
        )
        return res
