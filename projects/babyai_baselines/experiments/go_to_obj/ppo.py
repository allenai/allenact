from projects.babyai_baselines.experiments.go_to_obj.base import (
    BaseBabyAIGoToObjExperimentConfig,
)
from utils.experiment_utils import PipelineStage


class PPOBabyAIGoToObjExperimentConfig(BaseBabyAIGoToObjExperimentConfig):
    """PPO only."""

    @classmethod
    def tag(cls):
        return "BabyAIGoToObjPPO"

    @classmethod
    def training_pipeline(cls, **kwargs):
        total_train_steps = cls.TOTAL_RL_TRAIN_STEPS
        ppo_info = cls.rl_loss_default("ppo", steps=total_train_steps)

        return cls._training_pipeline(
            named_losses={"ppo_loss": ppo_info["loss"],},
            pipeline_stages=[
                PipelineStage(
                    loss_names=["ppo_loss"], max_stage_steps=total_train_steps,
                ),
            ],
            num_mini_batch=ppo_info["num_mini_batch"],
            update_repeats=ppo_info["update_repeats"],
            total_train_steps=total_train_steps,
        )
