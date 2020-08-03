from projects.babyai_baselines.experiments.go_to_local.base import (
    BaseBabyAIGoToLocalExperimentConfig,
)
from utils.experiment_utils import PipelineStage, LinearDecay


class DaggerBabyAIGoToLocalExperimentConfig(BaseBabyAIGoToLocalExperimentConfig):
    """Find goal in lighthouse env using imitation learning.

    Training with Dagger.
    """

    USE_EXPERT = True

    @classmethod
    def tag(cls):
        return "BabyAIGoToLocalDagger"

    @classmethod
    def training_pipeline(cls, **kwargs):
        total_train_steps = cls.TOTAL_IL_TRAIN_STEPS
        loss_info = cls.rl_loss_default("imitation")
        return cls._training_pipeline(
            named_losses={"imitation_loss": loss_info["loss"]},
            pipeline_stages=[
                PipelineStage(
                    loss_names=["imitation_loss"],
                    teacher_forcing=LinearDecay(
                        startp=1.0, endp=0.0, steps=total_train_steps // 2,
                    ),
                    max_stage_steps=total_train_steps,
                )
            ],
            num_mini_batch=loss_info["num_mini_batch"],
            update_repeats=loss_info["update_repeats"],
            total_train_steps=total_train_steps,
        )
