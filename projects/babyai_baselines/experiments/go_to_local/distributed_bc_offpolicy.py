import os

import torch

from .bc_offpolicy import BCOffPolicyBabyAIGoToLocalExperimentConfig
from plugins.babyai_plugin.babyai_constants import BABYAI_EXPERT_TRAJECTORIES_DIR
from plugins.minigrid_plugin.minigrid_offpolicy import (
    MiniGridOffPolicyExpertCELoss,
    create_minigrid_offpolicy_data_iterator,
)
from utils.experiment_utils import PipelineStage, OffPolicyPipelineComponent


class DistributedBCOffPolicyBabyAIGoToLocalExperimentConfig(
    BCOffPolicyBabyAIGoToLocalExperimentConfig
):
    """Distributed Off policy imitation."""

    @classmethod
    def tag(cls):
        return "DistributedBabyAIGoToLocalBCOffPolicy"

    @classmethod
    def machine_params(
        cls, mode="train", gpu_id="default", n_train_processes="default", **kwargs
    ):
        res = super().machine_params(mode, gpu_id, n_train_processes, **kwargs)

        if (
            res["nprocesses"] > 0
            and isinstance(res["gpu_ids"][0], int)
            and res["gpu_ids"][0] >= 0
        ):
            res["nprocesses"] = [res["nprocesses"] // 2] * 2
            res["gpu_ids"] = [0, 1]

        return res

    @classmethod
    def expert_ce_loss_kwargs_generator(cls):
        return dict(num_workers=len(cls.machine_params("train")["gpu_ids"]))

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
                    // cls.expert_ce_loss_kwargs_generator()["num_workers"]
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
                        data_iterator_kwargs_generator=cls.expert_ce_loss_kwargs_generator,
                        loss_names=["offpolicy_expert_ce_loss"],
                        updates=num_mini_batch * update_repeats,
                    ),
                ),
            ],
            num_mini_batch=0,
            update_repeats=0,
            total_train_steps=total_train_steps,
        )
