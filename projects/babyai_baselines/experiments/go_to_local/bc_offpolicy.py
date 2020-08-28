import os
from typing import Optional, List, Tuple

import torch
from gym_minigrid.minigrid import MiniGridEnv

from plugins.babyai_plugin.babyai_constants import BABYAI_EXPERT_TRAJECTORIES_DIR
from plugins.minigrid_plugin.minigrid_offpolicy import (
    MiniGridOffPolicyExpertCELoss,
    create_minigrid_offpolicy_data_iterator,
)
from projects.babyai_baselines.experiments.go_to_local.base import (
    BaseBabyAIGoToLocalExperimentConfig,
)
from utils.experiment_utils import PipelineStage, OffPolicyPipelineComponent


class BCOffPolicyBabyAIGoToLocalExperimentConfig(BaseBabyAIGoToLocalExperimentConfig):
    """BC Off policy imitation."""

    DATASET: Optional[List[Tuple[str, bytes, List[int], MiniGridEnv.Actions]]] = None

    GPU_ID = 0 if torch.cuda.is_available() else None

    def __init__(self):
        super().__init__()

    @classmethod
    def tag(cls):
        return "BabyAIGoToLocalBCOffPolicy"

    @classmethod
    def METRIC_ACCUMULATE_INTERVAL(cls):
        return 1

    @classmethod
    def training_pipeline(cls, **kwargs):
        total_train_steps = cls.TOTAL_IL_TRAIN_STEPS
        ppo_info = cls.rl_loss_default("ppo", steps=-1)

        num_mini_batch = ppo_info["num_mini_batch"]
        update_repeats = ppo_info["update_repeats"]

        return cls._training_pipeline(
            named_losses={
                "offpolicy_expert_ce_loss": MiniGridOffPolicyExpertCELoss(
                    total_episodes_in_epoch=int(1e6)
                ),
            },
            pipeline_stages=[
                PipelineStage(
                    loss_names=[],
                    max_stage_steps=total_train_steps,
                    offpolicy_component=OffPolicyPipelineComponent(
                        data_iterator_builder=lambda **kwargs: create_minigrid_offpolicy_data_iterator(
                            path=os.path.join(
                                BABYAI_EXPERT_TRAJECTORIES_DIR,
                                "BabyAI-GoToLocal-v0{}.pkl".format(
                                    "" if torch.cuda.is_available() else "-small"
                                ),
                            ),
                            nrollouts=cls.NUM_TRAIN_SAMPLERS // num_mini_batch,
                            rollout_len=cls.ROLLOUT_STEPS,
                            instr_len=cls.INSTR_LEN,
                            **kwargs,
                        ),
                        loss_names=["offpolicy_expert_ce_loss"],
                        updates=num_mini_batch * update_repeats,
                    ),
                ),
            ],
            num_mini_batch=0,
            update_repeats=0,
            total_train_steps=total_train_steps,
        )
