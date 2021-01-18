import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

from allenact.algorithms.onpolicy_sync.losses.imitation import Imitation
from allenact.base_abstractions.sensor import ExpertActionSensor
from allenact.utils.experiment_utils import Builder, PipelineStage, TrainingPipeline, LinearDecay
from allenact_plugins.habitat_plugin.habitat_sensors import (
    RGBSensorHabitat,
    TargetCoordinatesSensorHabitat,
)
from allenact_plugins.habitat_plugin.habitat_tasks import PointNavTask
from projects.pointnav_baselines.experiments.habitat.debug_pointnav_habitat_base import (
    DebugPointNavHabitatBaseConfig,
)
from projects.pointnav_baselines.experiments.pointnav_mixin_simpleconvgru import (
    PointNavMixInSimpleConvGRUConfig,
)


class PointNavHabitatRGBDeterministiSimpleConvGRUImitationExperimentConfig(
    DebugPointNavHabitatBaseConfig, PointNavMixInSimpleConvGRUConfig
):
    """An Point Navigation experiment configuration in Habitat with Depth
    input."""

    SENSORS = [
        RGBSensorHabitat(
            height=DebugPointNavHabitatBaseConfig.SCREEN_SIZE,
            width=DebugPointNavHabitatBaseConfig.SCREEN_SIZE,
            use_resnet_normalization=True,
        ),
        TargetCoordinatesSensorHabitat(coordinate_dims=2),
        ExpertActionSensor(nactions=len(PointNavTask.class_action_names())),
    ]

    @classmethod
    def tag(cls):
        return "Debug-Pointnav-Habitat-RGB-SimpleConv-BC"

    @classmethod
    def training_pipeline(cls, **kwargs):
        imitate_steps = int(75000000)
        lr = 3e-4
        num_mini_batch = 1
        update_repeats = 3
        num_steps = 30
        save_interval = 5000000
        log_interval = 10000 if torch.cuda.is_available() else 1
        gamma = 0.99
        use_gae = True
        gae_lambda = 0.95
        max_grad_norm = 0.5
        return TrainingPipeline(
            save_interval=save_interval,
            metric_accumulate_interval=log_interval,
            optimizer_builder=Builder(optim.Adam, dict(lr=lr)),
            num_mini_batch=num_mini_batch,
            update_repeats=update_repeats,
            max_grad_norm=max_grad_norm,
            num_steps=num_steps,
            named_losses={"imitation_loss": Imitation()},
            gamma=gamma,
            use_gae=use_gae,
            gae_lambda=gae_lambda,
            advance_scene_rollout_period=cls.ADVANCE_SCENE_ROLLOUT_PERIOD,
            pipeline_stages=[
                PipelineStage(
                    loss_names=["imitation_loss"],
                    max_stage_steps=imitate_steps,
                    # teacher_forcing=LinearDecay(steps=int(1e5), startp=1.0, endp=0.0,),
                ),
            ],
            lr_scheduler_builder=Builder(
                LambdaLR, {"lr_lambda": LinearDecay(steps=imitate_steps)}
            ),
        )
