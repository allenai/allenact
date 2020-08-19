from typing import Dict, Any, List, Optional
import abc

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torchvision import models

from core.base_abstractions.experiment_config import ExperimentConfig
from utils.experiment_utils import Builder, PipelineStage, TrainingPipeline, LinearDecay
from core.algorithms.onpolicy_sync.losses import PPO
from core.algorithms.onpolicy_sync.losses.ppo import PPOConfig
from core.base_abstractions.preprocessor import ObservationSet
from plugins.habitat_plugin.habitat_preprocessors import ResnetPreProcessorHabitat
from utils.viz_utils import (
    SimpleViz,
    TrajectoryViz,
    ActorViz,
    AgentViewViz,
    TensorViz1D,
    TensorViz2D,
)
from plugins.robothor_plugin.robothor_viz import ThorViz


class Resnet18NavBaseConfig(ExperimentConfig, abc.ABC):
    """An Object Navigation base configuration in RoboTHOR."""

    TRAIN_SCENES = [
        "FloorPlan_Train%d_%d" % (wall + 1, furniture + 1)
        for wall in range(12)
        for furniture in range(5)
    ][:1]

    # VALID_SCENES = [
    #     "FloorPlan_Val%d_%d" % (wall + 1, furniture + 1)
    #     for wall in range(3)
    #     for furniture in range(5)
    # ]
    VALID_SCENES = TRAIN_SCENES

    # TEST_SCENES = [
    #     "FloorPlan_test-dev%d_%d" % (wall + 1, furniture + 1)
    #     for wall in range(2)
    #     for furniture in range(2)
    # ]

    MAX_STEPS: int
    ADVANCE_SCENE_ROLLOUT_PERIOD = 10000000000000  # if more than 1 scene per worker

    VALIDATION_SAMPLES_PER_SCENE = 16

    NUM_PROCESSES = 56 if torch.cuda.is_available() else 4

    VISION_UUID = "dummy"
    TARGET_UUID = "dummy"
    RESNET_OUTPUT_UUID = "resnet"

    ENV_ARGS: Dict

    def __init__(self):
        self.PREPROCESSORS = [
            Builder(
                ResnetPreProcessorHabitat,
                dict(
                    input_height=self.SCREEN_SIZE,
                    input_width=self.SCREEN_SIZE,
                    output_width=7,
                    output_height=7,
                    output_dims=512,
                    pool=False,
                    torchvision_resnet_model=models.resnet18,
                    input_uuids=[self.VISION_UUID],
                    output_uuid=self.RESNET_OUTPUT_UUID,
                    parallel=False,
                ),
            ),
        ]

    @classmethod
    def training_pipeline(cls, **kwargs):
        ppo_steps = int(1e6)
        return TrainingPipeline(
            save_interval=200000,
            metric_accumulate_interval=1,
            optimizer_builder=Builder(optim.Adam, dict(lr=3e-4)),
            num_mini_batch=2,
            update_repeats=3,
            max_grad_norm=0.5,
            num_steps=30,
            named_losses={"ppo_loss": Builder(PPO, kwargs={}, default=PPOConfig,)},
            gamma=0.99,
            use_gae=True,
            gae_lambda=0.95,
            advance_scene_rollout_period=cls.ADVANCE_SCENE_ROLLOUT_PERIOD,
            pipeline_stages=[
                PipelineStage(loss_names=["ppo_loss"], max_stage_steps=ppo_steps)
            ],
            lr_scheduler_builder=Builder(
                LambdaLR, {"lr_lambda": LinearDecay(steps=ppo_steps)}
            ),
        )

    def split_num_processes(self, ndevices, nprocesses=None):
        if nprocesses is None:
            nprocesses = self.NUM_PROCESSES
        assert nprocesses >= ndevices, "NUM_PROCESSES {} < ndevices {}".format(
            nprocesses, ndevices
        )
        res = [0] * ndevices
        for it in range(nprocesses):
            res[it % ndevices] += 1
        return res

    def machine_params(self, mode="train", **kwargs):
        visualizer = None
        if mode == "train":
            gpu_ids = (
                ["cpu"]
                if not torch.cuda.is_available()
                else list(range(torch.cuda.device_count()))
            )
            nprocesses = self.split_num_processes(len(gpu_ids))
        elif mode == "valid":
            gpu_ids = (
                ["cpu"]
                if not torch.cuda.is_available()
                else [torch.cuda.device_count() - 1]
            )
            nprocesses = 1
        elif mode == "test":
            gpu_ids = (
                ["cpu"]
                if not torch.cuda.is_available()
                else list(range(torch.cuda.device_count()))
            )
            nprocesses = self.split_num_processes(len(gpu_ids))

            # visualizer = Builder(
            #     SimpleViz,
            #     dict(
            #         episode_ids=self.ep_ids,
            #         mode="test",
            #         # v1=Builder(TrajectoryViz, dict()),
            #         v3=Builder(ActorViz, dict(figsize=(3.25, 10), fontsize=(18))),
            #         # v4=Builder(TensorViz1D, dict()),
            #         # v5=Builder(TensorViz1D, dict(rollout_source=("masks"))),
            #         # v6=Builder(TensorViz2D, dict()),
            #         v7=Builder(
            #             ThorViz, dict(figsize=(16, 8), viz_rows_cols=(448, 448))
            #         ),
            #         v2=Builder(
            #             AgentViewViz,
            #             dict(max_video_length=100, episode_ids=self.video_ids),
            #         ),
            #     ),
            # )
        else:
            raise NotImplementedError("mode must be 'train', 'valid', or 'test'.")

        observation_set = (
            Builder(
                ObservationSet,
                kwargs=dict(
                    source_ids=self.OBSERVATIONS,
                    all_preprocessors=self.PREPROCESSORS,
                    all_sensors=self.SENSORS,
                ),
            )
            if (isinstance(nprocesses, int) and nprocesses > 0)
            or (isinstance(nprocesses, List) and max(nprocesses) > 0)
            else None
        )

        return {
            "nprocesses": nprocesses,
            "gpu_ids": gpu_ids,
            "observation_set": observation_set,
            "visualizer": visualizer,
        }

    @abc.abstractmethod
    def _get_sampler_args_for_scene_split(
        self,
        scenes: List[str],
        process_ind: int,
        total_processes: int,
        seeds: Optional[List[int]] = None,
        deterministic_cudnn: bool = False,
    ) -> Dict[str, Any]:
        raise NotImplementedError()

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
            ("0.%d" % devices[process_ind % len(devices)])
            if devices is not None and len(devices) > 0
            else None
        )
        return res

    def test_task_sampler_args(
        self,
        process_ind: int,
        total_processes: int,
        devices: Optional[List[int]] = None,
        seeds: Optional[List[int]] = None,
        deterministic_cudnn: bool = False,
    ) -> Dict[str, Any]:
        return self.valid_task_sampler_args(
            process_ind, total_processes, devices, seeds, deterministic_cudnn
        )
