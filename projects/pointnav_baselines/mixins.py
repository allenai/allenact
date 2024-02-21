from typing import Optional
from typing import Sequence

import attr
import gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

from allenact.algorithms.onpolicy_sync.losses import PPO
from allenact.algorithms.onpolicy_sync.losses.ppo import PPOConfig
from allenact.base_abstractions.sensor import Sensor
from allenact.embodiedai.sensors.vision_sensors import RGBSensor, DepthSensor
from allenact.utils.experiment_utils import (
    Builder,
    TrainingPipeline,
    PipelineStage,
    LinearDecay,
)
from projects.objectnav_baselines.mixins import update_with_auxiliary_losses

# fmt: off
try:
    # Habitat may not be installed, just create a fake class here in that case
    from allenact_plugins.habitat_plugin.habitat_sensors import TargetCoordinatesSensorHabitat
except ImportError:
    class TargetCoordinatesSensorHabitat:  #type:ignore
        pass
# fmt: on

from allenact_plugins.robothor_plugin.robothor_sensors import GPSCompassSensorRoboThor
from allenact_plugins.robothor_plugin.robothor_tasks import PointNavTask
from allenact_plugins.navigation_plugin.pointnav.models import PointNavActorCritic


@attr.s(kw_only=True)
class PointNavUnfrozenResNetWithGRUActorCriticMixin:
    backbone: str = attr.ib()
    sensors: Sequence[Sensor] = attr.ib()
    auxiliary_uuids: Sequence[str] = attr.ib()
    add_prev_actions: bool = attr.ib()
    multiple_beliefs: bool = attr.ib()
    belief_fusion: Optional[str] = attr.ib()

    def create_model(self, **kwargs) -> nn.Module:
        rgb_uuid = next(
            (s.uuid for s in self.sensors if isinstance(s, RGBSensor)), None
        )
        depth_uuid = next(
            (s.uuid for s in self.sensors if isinstance(s, DepthSensor)), None
        )
        goal_sensor_uuid = next(
            (
                s.uuid
                for s in self.sensors
                if isinstance(
                    s, (GPSCompassSensorRoboThor, TargetCoordinatesSensorHabitat)
                )
            )
        )

        return PointNavActorCritic(
            # Env and Tak
            action_space=gym.spaces.Discrete(len(PointNavTask.class_action_names())),
            observation_space=kwargs["sensor_preprocessor_graph"].observation_spaces,
            rgb_uuid=rgb_uuid,
            depth_uuid=depth_uuid,
            goal_sensor_uuid=goal_sensor_uuid,
            # RNN
            hidden_size=(
                228 if self.multiple_beliefs and len(self.auxiliary_uuids) > 1 else 512
            ),
            num_rnn_layers=1,
            rnn_type="GRU",
            add_prev_actions=self.add_prev_actions,
            action_embed_size=4,
            # CNN
            backbone=self.backbone,
            resnet_baseplanes=32,
            embed_coordinates=False,
            coordinate_dims=2,
            # Aux
            auxiliary_uuids=self.auxiliary_uuids,
            multiple_beliefs=self.multiple_beliefs,
            beliefs_fusion=self.belief_fusion,
        )


class PointNavPPOMixin:
    @staticmethod
    def training_pipeline(
        auxiliary_uuids: Sequence[str],
        multiple_beliefs: bool,
        normalize_advantage: bool,
        advance_scene_rollout_period: Optional[int] = None,
    ) -> TrainingPipeline:
        ppo_steps = int(75000000)
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

        named_losses = {
            "ppo_loss": (PPO(**PPOConfig, normalize_advantage=normalize_advantage), 1.0)
        }
        named_losses = update_with_auxiliary_losses(
            named_losses=named_losses,
            auxiliary_uuids=auxiliary_uuids,
            multiple_beliefs=multiple_beliefs,
        )

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
            advance_scene_rollout_period=advance_scene_rollout_period,
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
