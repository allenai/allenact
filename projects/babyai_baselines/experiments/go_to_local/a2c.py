import torch

from projects.babyai_baselines.experiments.go_to_local.base import BaseBabyAIGoToLocalExperimentConfig
from utils.experiment_utils import PipelineStage


class A2CBabyAIGoToLocalExperimentConfig(BaseBabyAIGoToLocalExperimentConfig):
    """A2C only."""

    NUM_TRAIN_SAMPLERS: int = 128 * 6 if torch.cuda.is_available() else 4
    ROLLOUT_STEPS: int = 16

    @classmethod
    def tag(cls):
        return "BabyAIGoToLocalA2C"

    @classmethod
    def training_pipeline(cls, **kwargs):
        total_training_steps = cls.TOTAL_RL_TRAIN_STEPS
        a2c_info = cls.rl_loss_default("a2c", steps=total_training_steps)

        return cls._training_pipeline(
            named_losses={"a2c_loss": a2c_info["loss"],},
            pipeline_stages=[
                PipelineStage(loss_names=["a2c_loss"], max_stage_steps=total_training_steps,),
            ],
            num_mini_batch=a2c_info["num_mini_batch"],
            update_repeats=a2c_info["update_repeats"],
            total_train_steps=total_training_steps,
        )
