from typing import Dict, Tuple

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

from allenact.algorithms.onpolicy_sync.losses import PPO
from allenact.algorithms.onpolicy_sync.losses.abstract_loss import (
    AbstractActorCriticLoss,
)
from allenact.algorithms.onpolicy_sync.losses.ppo import PPOConfig
from allenact.embodiedai.aux_losses.losses import (
    MultiAuxTaskNegEntropyLoss,
    InverseDynamicsLoss,
    TemporalDistanceLoss,
    CPCA1Loss,
    CPCA2Loss,
    CPCA4Loss,
    CPCA8Loss,
    CPCA16Loss,
)

# noinspection PyUnresolvedReferences
from allenact.embodiedai.models.fusion_models import (
    AverageFusion,
    SoftmaxFusion,
    AttentiveFusion,
)
from allenact.utils.experiment_utils import (
    Builder,
    PipelineStage,
    TrainingPipeline,
    LinearDecay,
)
from projects.objectnav_baselines.experiments.objectnav_base import ObjectNavBaseConfig


class ObjectNavMixInPPOConfig(ObjectNavBaseConfig):
    # selected auxiliary uuids
    ## if comment all the keys, then it's vanilla DD-PPO
    AUXILIARY_UUIDS = [
        # InverseDynamicsLoss.UUID,
        # TemporalDistanceLoss.UUID,
        # CPCA1Loss.UUID,
        # CPCA4Loss.UUID,
        # CPCA8Loss.UUID,
        # CPCA16Loss.UUID,
    ]

    ADD_PREV_ACTIONS = False

    MULTIPLE_BELIEFS = False
    BELIEF_FUSION = (  # choose one
        None
        # AttentiveFusion
        # AverageFusion
        # SoftmaxFusion
    )

    def training_pipeline(self, **kwargs):
        # PPO
        ppo_steps = int(300000000)
        lr = 3e-4
        num_mini_batch = 1
        update_repeats = 4
        num_steps = 128
        save_interval = 5000000
        log_interval = 10000 if torch.cuda.is_available() else 1
        gamma = 0.99
        use_gae = True
        gae_lambda = 0.95
        max_grad_norm = 0.5

        named_losses = {"ppo_loss": (PPO(**PPOConfig), 1.0)}
        named_losses = self._update_with_auxiliary_losses(named_losses)

        return TrainingPipeline(
            save_interval=save_interval,
            metric_accumulate_interval=log_interval,
            optimizer_builder=Builder(optim.Adam, dict(lr=lr)),
            num_mini_batch=num_mini_batch,
            update_repeats=update_repeats,
            max_grad_norm=max_grad_norm,
            num_steps=num_steps,
            named_losses={key: val[0] for key, val in named_losses.items()},
            gamma=gamma,
            use_gae=use_gae,
            gae_lambda=gae_lambda,
            advance_scene_rollout_period=self.ADVANCE_SCENE_ROLLOUT_PERIOD,
            pipeline_stages=[
                PipelineStage(
                    loss_names=list(named_losses.keys()),
                    max_stage_steps=ppo_steps,
                    loss_weights=[val[1] for val in named_losses.values()],
                )
            ],
            lr_scheduler_builder=Builder(
                LambdaLR, {"lr_lambda": LinearDecay(steps=ppo_steps)}
            ),
        )

    @classmethod
    def _update_with_auxiliary_losses(cls, named_losses):
        # auxliary losses
        aux_loss_total_weight = 2.0

        # Total losses
        total_aux_losses: Dict[str, Tuple[AbstractActorCriticLoss, float]] = {
            InverseDynamicsLoss.UUID: (
                InverseDynamicsLoss(
                    subsample_rate=0.2, subsample_min_num=10,  # TODO: test its effects
                ),
                0.05 * aux_loss_total_weight,  # should times 2
            ),
            TemporalDistanceLoss.UUID: (
                TemporalDistanceLoss(
                    num_pairs=8, epsiode_len_min=5,  # TODO: test its effects
                ),
                0.2 * aux_loss_total_weight,  # should times 2
            ),
            CPCA1Loss.UUID: (
                CPCA1Loss(subsample_rate=0.2,),  # TODO: test its effects
                0.05 * aux_loss_total_weight,  # should times 2
            ),
            CPCA2Loss.UUID: (
                CPCA2Loss(subsample_rate=0.2,),  # TODO: test its effects
                0.05 * aux_loss_total_weight,  # should times 2
            ),
            CPCA4Loss.UUID: (
                CPCA4Loss(subsample_rate=0.2,),  # TODO: test its effects
                0.05 * aux_loss_total_weight,  # should times 2
            ),
            CPCA8Loss.UUID: (
                CPCA8Loss(subsample_rate=0.2,),  # TODO: test its effects
                0.05 * aux_loss_total_weight,  # should times 2
            ),
            CPCA16Loss.UUID: (
                CPCA16Loss(subsample_rate=0.2,),  # TODO: test its effects
                0.05 * aux_loss_total_weight,  # should times 2
            ),
        }
        named_losses.update(
            {uuid: total_aux_losses[uuid] for uuid in cls.AUXILIARY_UUIDS}
        )

        if cls.MULTIPLE_BELIEFS:  # add weight entropy loss automatically
            named_losses[MultiAuxTaskNegEntropyLoss.UUID] = (
                MultiAuxTaskNegEntropyLoss(cls.AUXILIARY_UUIDS),
                0.01,
            )

        return named_losses
