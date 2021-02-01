from abc import ABC
from typing import Dict, Any, List, Optional, Union, Sequence, cast

import gym
import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import LambdaLR

from allenact.algorithms.onpolicy_sync.losses import PPO, A2C
from allenact.algorithms.onpolicy_sync.losses.a2cacktr import A2CConfig
from allenact.algorithms.onpolicy_sync.losses.imitation import Imitation
from allenact.algorithms.onpolicy_sync.losses.ppo import PPOConfig
from allenact.base_abstractions.experiment_config import ExperimentConfig, MachineParams
from allenact.base_abstractions.misc import Loss
from allenact.base_abstractions.sensor import SensorSuite, Sensor, ExpertActionSensor
from allenact.base_abstractions.task import TaskSampler
from allenact.utils.experiment_utils import (
    Builder,
    LinearDecay,
    PipelineStage,
    TrainingPipeline,
)
from allenact_plugins.babyai_plugin.babyai_models import BabyAIRecurrentACModel
from allenact_plugins.babyai_plugin.babyai_tasks import BabyAITask, BabyAITaskSampler
from allenact_plugins.minigrid_plugin.minigrid_sensors import (
    EgocentricMiniGridSensor,
    MiniGridMissionSensor,
)


class BaseBabyAIExperimentConfig(ExperimentConfig, ABC):
    """Base experimental config."""

    LEVEL: Optional[str] = None
    TOTAL_RL_TRAIN_STEPS: Optional[int] = None
    AGENT_VIEW_SIZE: int = 7
    ROLLOUT_STEPS: Optional[int] = None
    NUM_TRAIN_SAMPLERS: Optional[int] = None
    NUM_TEST_TASKS: Optional[int] = None
    INSTR_LEN: Optional[int] = None
    USE_INSTR: Optional[bool] = None
    GPU_ID: Optional[int] = None
    USE_EXPERT = False
    SHOULD_LOG = True
    PPO_NUM_MINI_BATCH = 2
    ARCH: Optional[str] = None
    NUM_CKPTS_TO_SAVE = 50

    TEST_SEED_OFFSET = 0

    DEFAULT_LR = 1e-3

    @classmethod
    def METRIC_ACCUMULATE_INTERVAL(cls):
        return cls.NUM_TRAIN_SAMPLERS * 1000

    @classmethod
    def get_sensors(cls) -> Sequence[Sensor]:
        assert cls.USE_INSTR is not None

        return (
            [
                EgocentricMiniGridSensor(
                    agent_view_size=cls.AGENT_VIEW_SIZE, view_channels=3
                ),
            ]
            + (
                [MiniGridMissionSensor(instr_len=cls.INSTR_LEN)]  # type:ignore
                if cls.USE_INSTR
                else []
            )
            + (
                [
                    ExpertActionSensor(  # type: ignore
                        nactions=len(BabyAITask.class_action_names())
                    )
                ]
                if cls.USE_EXPERT
                else []
            )
        )

    @classmethod
    def rl_loss_default(cls, alg: str, steps: Optional[int] = None):
        if alg == "ppo":
            assert steps is not None
            return {
                "loss": Builder(
                    PPO, kwargs={"clip_decay": LinearDecay(steps)}, default=PPOConfig,
                ),
                "num_mini_batch": cls.PPO_NUM_MINI_BATCH,
                "update_repeats": 4,
            }
        elif alg == "a2c":
            return {
                "loss": A2C(**A2CConfig),
                "num_mini_batch": 1,
                "update_repeats": 1,
            }
        elif alg == "imitation":
            return {
                "loss": Imitation(),
                "num_mini_batch": cls.PPO_NUM_MINI_BATCH,
                "update_repeats": 4,
            }
        else:
            raise NotImplementedError

    @classmethod
    def _training_pipeline(
        cls,
        named_losses: Dict[str, Union[Loss, Builder]],
        pipeline_stages: List[PipelineStage],
        num_mini_batch: int,
        update_repeats: int,
        total_train_steps: int,
        lr: Optional[float] = None,
    ):
        lr = cls.DEFAULT_LR if lr is None else lr

        num_steps = cls.ROLLOUT_STEPS
        metric_accumulate_interval = (
            cls.METRIC_ACCUMULATE_INTERVAL()
        )  # Log every 10 max length tasks
        save_interval = int(cls.TOTAL_RL_TRAIN_STEPS / cls.NUM_CKPTS_TO_SAVE)
        gamma = 0.99

        use_gae = "reinforce_loss" not in named_losses
        gae_lambda = 0.99
        max_grad_norm = 0.5

        return TrainingPipeline(
            save_interval=save_interval,
            metric_accumulate_interval=metric_accumulate_interval,
            optimizer_builder=Builder(cast(optim.Optimizer, optim.Adam), dict(lr=lr)),
            num_mini_batch=num_mini_batch,
            update_repeats=update_repeats,
            max_grad_norm=max_grad_norm,
            num_steps=num_steps,
            named_losses=named_losses,
            gamma=gamma,
            use_gae=use_gae,
            gae_lambda=gae_lambda,
            advance_scene_rollout_period=None,
            should_log=cls.SHOULD_LOG,
            pipeline_stages=pipeline_stages,
            lr_scheduler_builder=Builder(
                LambdaLR, {"lr_lambda": LinearDecay(steps=cls.TOTAL_RL_TRAIN_STEPS)}  # type: ignore
            ),
        )

    @classmethod
    def machine_params(
        cls, mode="train", gpu_id="default", n_train_processes="default", **kwargs
    ):
        if mode == "train":
            if n_train_processes == "default":
                nprocesses = cls.NUM_TRAIN_SAMPLERS
            else:
                nprocesses = n_train_processes
        elif mode == "valid":
            nprocesses = 0
        elif mode == "test":
            nprocesses = min(
                100 if torch.cuda.is_available() else 8, cls.NUM_TEST_TASKS
            )
        else:
            raise NotImplementedError("mode must be 'train', 'valid', or 'test'.")

        if gpu_id == "default":
            devices = [] if cls.GPU_ID is None else [cls.GPU_ID]
        else:
            devices = [gpu_id]

        return MachineParams(nprocesses=nprocesses, devices=devices)

    @classmethod
    def create_model(cls, **kwargs) -> nn.Module:
        sensors = cls.get_sensors()
        return BabyAIRecurrentACModel(
            action_space=gym.spaces.Discrete(len(BabyAITask.class_action_names())),
            observation_space=SensorSuite(sensors).observation_spaces,
            use_instr=cls.USE_INSTR,
            use_memory=True,
            arch=cls.ARCH,
        )

    @classmethod
    def make_sampler_fn(cls, **kwargs) -> TaskSampler:
        return BabyAITaskSampler(**kwargs)

    def train_task_sampler_args(
        self,
        process_ind: int,
        total_processes: int,
        devices: Optional[List[int]] = None,
        seeds: Optional[List[int]] = None,
        deterministic_cudnn: bool = False,
    ) -> Dict[str, Any]:
        return {
            "env_builder": self.LEVEL,
            "sensors": self.get_sensors(),
            "seed": seeds[process_ind] if seeds is not None else None,
        }

    def valid_task_sampler_args(
        self,
        process_ind: int,
        total_processes: int,
        devices: Optional[List[int]] = None,
        seeds: Optional[List[int]] = None,
        deterministic_cudnn: bool = False,
    ) -> Dict[str, Any]:
        raise RuntimeError

    def test_task_sampler_args(
        self,
        process_ind: int,
        total_processes: int,
        devices: Optional[List[int]] = None,
        seeds: Optional[List[int]] = None,
        deterministic_cudnn: bool = False,
    ) -> Dict[str, Any]:
        max_tasks = self.NUM_TEST_TASKS // total_processes + (
            process_ind < (self.NUM_TEST_TASKS % total_processes)
        )
        task_seeds_list = [
            2 ** 31 - 1 + self.TEST_SEED_OFFSET + process_ind + total_processes * i
            for i in range(max_tasks)
        ]
        # print(max_tasks, process_ind, total_processes, task_seeds_list)

        assert len(task_seeds_list) == 0 or (
            min(task_seeds_list) >= 0 and max(task_seeds_list) <= 2 ** 32 - 1
        )

        train_sampler_args = self.train_task_sampler_args(
            process_ind=process_ind,
            total_processes=total_processes,
            devices=devices,
            seeds=seeds,
            deterministic_cudnn=deterministic_cudnn,
        )
        return {
            **train_sampler_args,
            "task_seeds_list": task_seeds_list,
            "max_tasks": max_tasks,
            "deterministic_sampling": True,
            "sensors": [
                s for s in train_sampler_args["sensors"] if "Expert" not in str(type(s))
            ],
        }
