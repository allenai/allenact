from allenact.utils.experiment_utils import PipelineStage
from projects.babyai_baselines.experiments.go_to_local.base import (
    BaseBabyAIGoToLocalExperimentConfig,
)


class PPOBabyAIGoToLocalExperimentConfig(BaseBabyAIGoToLocalExperimentConfig):
    """Behavior clone then PPO."""

    USE_EXPERT = True

    @classmethod
    def tag(cls):
        return "BabyAIGoToLocalBC"

    @classmethod
    def training_pipeline(cls, **kwargs):
        total_train_steps = cls.TOTAL_IL_TRAIN_STEPS

        ppo_info = cls.rl_loss_default("ppo", steps=-1)
        imitation_info = cls.rl_loss_default("imitation")

        return cls._training_pipeline(
            named_losses={
                "imitation_loss": imitation_info["loss"],
            },
            pipeline_stages=[
                PipelineStage(
                    loss_names=["imitation_loss"],
                    max_stage_steps=total_train_steps,
                ),
            ],
            num_mini_batch=min(
                info["num_mini_batch"] for info in [ppo_info, imitation_info]
            ),
            update_repeats=min(
                info["update_repeats"] for info in [ppo_info, imitation_info]
            ),
            total_train_steps=total_train_steps,
        )
