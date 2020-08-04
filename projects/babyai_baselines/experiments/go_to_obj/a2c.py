from projects.babyai_baselines.experiments.go_to_obj.base import (
    BaseBabyAIGoToObjExperimentConfig,
)
from utils.experiment_utils import PipelineStage


class A2CBabyAIGoToObjExperimentConfig(BaseBabyAIGoToObjExperimentConfig):
    """A2C only."""

    TOTAL_RL_TRAIN_STEPS = int(1e5)

    @classmethod
    def tag(cls):
        return "BabyAIGoToObjA2C"

    @classmethod
    def training_pipeline(cls, **kwargs):
        total_training_steps = cls.TOTAL_RL_TRAIN_STEPS
        a2c_info = cls.rl_loss_default("a2c", steps=total_training_steps)

        return cls._training_pipeline(
            named_losses={"a2c_loss": a2c_info["loss"],},
            pipeline_stages=[
                PipelineStage(
                    loss_names=["a2c_loss"], max_stage_steps=total_training_steps,
                ),
            ],
            num_mini_batch=a2c_info["num_mini_batch"],
            update_repeats=a2c_info["update_repeats"],
            total_train_steps=total_training_steps,
        )
