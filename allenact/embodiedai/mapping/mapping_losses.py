import torch
from torch.nn import functional as F

from allenact.algorithms.onpolicy_sync.losses.abstract_loss import (
    AbstractActorCriticLoss,
)
from allenact.algorithms.onpolicy_sync.policy import ObservationType
from allenact.base_abstractions.distributions import CategoricalDistr
from allenact.base_abstractions.misc import ActorCriticOutput


class BinnedPointCloudMapLoss(AbstractActorCriticLoss):
    """A (binary cross entropy) loss for training metric maps for free space
    prediction."""

    def __init__(
        self,
        binned_pc_uuid: str,
        map_logits_uuid: str,
    ):
        """Initializer.

        # Parameters
        binned_pc_uuid : The uuid of a sensor returning
            a dictionary with an "egocentric_update"
            key with the same format as returned by
            `allenact.embodied_ai.mapping_utils.map_builders.BinnedPointCloudMapBuilder`. Such a sensor
            can be found in the `allenact_plugins` library: see
            `allenact_plugins.ithor_plugin.ithor_sensors.BinnedPointCloudMapTHORSensor`.
        map_logits_uuid : key used to index into `actor_critic_output.extras` (returned by the model)
            whose value should be a tensor of the same shape as the tensor corresponding to the above
            "egocentric_update" key.
        """
        super().__init__()
        self.binned_pc_uuid = binned_pc_uuid
        self.map_logits_uuid = map_logits_uuid

    def loss(  # type: ignore
        self,
        step_count: int,
        batch: ObservationType,
        actor_critic_output: ActorCriticOutput[CategoricalDistr],
        *args,
        **kwargs,
    ):
        ego_map_gt = batch["observations"][self.binned_pc_uuid][
            "egocentric_update"
        ].float()
        *_, h, w, c = ego_map_gt.shape
        ego_map_gt = ego_map_gt.view(-1, h, w, c).permute(0, 3, 1, 2).contiguous()

        ego_map_logits = actor_critic_output.extras[self.map_logits_uuid]
        vision_range = ego_map_logits.shape[-1]
        ego_map_logits = ego_map_logits.view(-1, c, vision_range, vision_range)

        assert ego_map_gt.shape == ego_map_logits.shape

        ego_map_gt_thresholded = (ego_map_gt > 0.5).float()
        total_loss = F.binary_cross_entropy_with_logits(
            ego_map_logits, ego_map_gt_thresholded
        )

        return (
            total_loss,
            {"binned_pc_map_ce": total_loss.item()},
        )

        # FOR DEBUGGING: Save all the ground-truth & predicted maps side by side
        # import numpy as np
        # import imageio
        # for i in range(ego_map_gt_thresholded.shape[0]):
        #     a = ego_map_gt_thresholded[i].permute(1, 2, 0).flip(0).detach().numpy()
        #     b = torch.sigmoid(ego_map_logits)[i].permute(1, 2, 0).flip(0).detach().numpy()
        #
        #     imageio.imwrite(
        #         f"z_occupancy_maps/{i}.png",
        #         np.concatenate((a, 1 + 0 * a[:, :10], b), axis=1),
        #     )


class SemanticMapFocalLoss(AbstractActorCriticLoss):
    """A (focal-loss based) loss for training metric maps for free space
    prediction.

    As semantic maps tend to be quite sparse this loss uses the focal
    loss (https://arxiv.org/abs/1708.02002) rather than binary cross
    entropy (BCE). If the `gamma` parameter is 0.0 then this is just the
    normal BCE, larger values of `gamma` result less and less emphasis
    being paid to examples that are already well classified.
    """

    def __init__(
        self, semantic_map_uuid: str, map_logits_uuid: str, gamma: float = 2.0
    ):
        """Initializer.

        # Parameters
        semantic_map_uuid : The uuid of a sensor returning
            a dictionary with an "egocentric_update"
            key with the same format as returned by
            `allenact.embodied_ai.mapping_utils.map_builders.SemanticMapBuilder`. Such a sensor
            can be found in the `allenact_plugins` library: see
            `allenact_plugins.ithor_plugin.ithor_sensors.SemanticMapTHORSensor`.
        map_logits_uuid : key used to index into `actor_critic_output.extras` (returned by the model)
            whose value should be a tensor of the same shape as the tensor corresponding to the above
            "egocentric_update" key.
        """
        super().__init__()
        assert gamma >= 0, f"`gamma` (=={gamma}) must be >= 0"
        self.semantic_map_uuid = semantic_map_uuid
        self.map_logits_uuid = map_logits_uuid
        self.gamma = gamma

    def loss(  # type: ignore
        self,
        step_count: int,
        batch: ObservationType,
        actor_critic_output: ActorCriticOutput[CategoricalDistr],
        *args,
        **kwargs,
    ):
        ego_map_gt = batch["observations"][self.semantic_map_uuid]["egocentric_update"]
        ego_map_gt = (
            ego_map_gt.view(-1, *ego_map_gt.shape[-3:]).permute(0, 3, 1, 2).contiguous()
        )

        ego_map_logits = actor_critic_output.extras[self.map_logits_uuid]
        ego_map_logits = ego_map_logits.view(-1, *ego_map_logits.shape[-3:])

        assert ego_map_gt.shape == ego_map_logits.shape

        p = torch.sigmoid(ego_map_logits)
        one_minus_p = torch.sigmoid(-ego_map_logits)

        log_p = F.logsigmoid(ego_map_logits)
        log_one_minus_p = F.logsigmoid(-ego_map_logits)

        ego_map_gt = ego_map_gt.float()
        total_loss = -(
            ego_map_gt * (log_p * (one_minus_p**self.gamma))
            + (1 - ego_map_gt) * (log_one_minus_p * (p**self.gamma))
        ).mean()

        return (
            total_loss,
            {"sem_map_focal_loss": total_loss.item()},
        )

        # FOR DEBUGGING: Save all the ground-truth & predicted maps side by side
        # import numpy as np
        # import imageio
        # from allenact.embodiedai.mapping.mapping_utils.map_builders import SemanticMapBuilder
        #
        # print("\n" * 3)
        # for i in range(ego_map_gt.shape[0]):
        #     pred_sem_map = torch.sigmoid(ego_map_logits)[i].permute(1, 2, 0).flip(0).detach()
        #     a = SemanticMapBuilder.randomly_color_semantic_map(ego_map_gt[i].permute(1, 2, 0).flip(0).detach())
        #     b = SemanticMapBuilder.randomly_color_semantic_map(pred_sem_map)
        #     imageio.imwrite(
        #         f"z_semantic_maps/{i}.png",
        #         np.concatenate((a, 255 + a[:, :10] * 0, b), axis=1),
        #     )
        #
