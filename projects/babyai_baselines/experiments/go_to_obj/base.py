from typing import Dict, List, Optional, Union

import gym
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import LambdaLR

from plugins.babyai_plugin.babyai_models import BabyAIRecurrentACModel
from plugins.babyai_plugin.babyai_tasks import BabyAITask
from projects.babyai_baselines.experiments.base import BaseBabyAIExperimentConfig
from core.base_abstractions.misc import Loss
from core.base_abstractions.sensor import SensorSuite
from utils.experiment_utils import Builder, LinearDecay, PipelineStage, TrainingPipeline


class BaseBabyAIGoToObjExperimentConfig(BaseBabyAIExperimentConfig):
    """Base experimental config."""

    LEVEL: Optional[str] = "BabyAI-GoToObj-v0"
    TOTAL_RL_TRAIN_STEPS = int(5e4)
    TOTAL_IL_TRAIN_STEPS = int(2e4)
    ROLLOUT_STEPS: int = 32
    NUM_TRAIN_SAMPLERS: int = 16
    PPO_NUM_MINI_BATCH = 2
    NUM_TEST_TASKS: int = 50
    USE_LR_DECAY: bool = False

    DEFAULT_LR = 1e-3

    ARCH = "cnn1"
    # ARCH = "cnn2"
    # ARCH = "expert_filmcnn"

    USE_INSTR = False
    INSTR_LEN: int = -1

    @classmethod
    def METRIC_ACCUMULATE_INTERVAL(cls):
        return cls.NUM_TRAIN_SAMPLERS * 128

    @classmethod
    def _training_pipeline(  # type:ignore
        cls,
        named_losses: Dict[str, Union[Loss, Builder]],
        pipeline_stages: List[PipelineStage],
        num_mini_batch: int,
        update_repeats: int,
        total_train_steps: int,
        lr: Optional[float] = None,
    ):
        lr = cls.DEFAULT_LR

        num_steps = cls.ROLLOUT_STEPS
        metric_accumulate_interval = (
            cls.METRIC_ACCUMULATE_INTERVAL()
        )  # Log every 10 max length tasks
        save_interval = 2 ** 31
        gamma = 0.99

        use_gae = "reinforce_loss" not in named_losses
        gae_lambda = 0.99
        max_grad_norm = 0.5

        return TrainingPipeline(
            save_interval=save_interval,
            metric_accumulate_interval=metric_accumulate_interval,
            optimizer_builder=Builder(optim.Adam, dict(lr=lr)),
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
                LambdaLR, {"lr_lambda": LinearDecay(steps=total_train_steps)}  # type: ignore
            )
            if cls.USE_LR_DECAY
            else None,
        )

    @classmethod
    def create_model(cls, **kwargs) -> nn.Module:
        sensors = cls.get_sensors()
        return BabyAIRecurrentACModel(
            action_space=gym.spaces.Discrete(len(BabyAITask.class_action_names())),
            observation_space=SensorSuite(sensors).observation_spaces,
            use_instr=cls.USE_INSTR,
            use_memory=True,
            arch=cls.ARCH,
            instr_dim=8,
            lang_model="gru",
            memory_dim=128,
        )
