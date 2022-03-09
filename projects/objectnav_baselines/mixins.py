from typing import Sequence, Union, Optional, Dict, Tuple, Type

import attr
import gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torchvision import models

from allenact.algorithms.onpolicy_sync.losses import PPO
from allenact.algorithms.onpolicy_sync.losses.abstract_loss import (
    AbstractActorCriticLoss,
)
from allenact.algorithms.onpolicy_sync.losses.imitation import Imitation
from allenact.algorithms.onpolicy_sync.losses.ppo import PPOConfig
from allenact.base_abstractions.preprocessor import Preprocessor
from allenact.base_abstractions.sensor import Sensor
from allenact.embodiedai.aux_losses.losses import (
    InverseDynamicsLoss,
    TemporalDistanceLoss,
    CPCA1Loss,
    CPCA2Loss,
    CPCA4Loss,
    CPCA8Loss,
    CPCA16Loss,
    MultiAuxTaskNegEntropyLoss,
)
from allenact.embodiedai.preprocessors.resnet import ResNetPreprocessor
from allenact.embodiedai.sensors.vision_sensors import RGBSensor, DepthSensor
from allenact.utils.experiment_utils import (
    Builder,
    TrainingPipeline,
    PipelineStage,
    LinearDecay,
)
from allenact_plugins.ithor_plugin.ithor_sensors import GoalObjectTypeThorSensor
from allenact_plugins.navigation_plugin.objectnav.models import (
    ResnetTensorNavActorCritic,
    ObjectNavActorCritic,
)
from allenact_plugins.robothor_plugin.robothor_tasks import ObjectNavTask


@attr.s(kw_only=True)
class ResNetPreprocessGRUActorCriticMixin:
    sensors: Sequence[Sensor] = attr.ib()
    resnet_type: str = attr.ib()
    screen_size: int = attr.ib()
    goal_sensor_type: Type[Sensor] = attr.ib()

    def preprocessors(self) -> Sequence[Union[Preprocessor, Builder[Preprocessor]]]:
        preprocessors = []

        if self.resnet_type in ["RN18", "RN34"]:
            output_shape = (512, 7, 7)
        elif self.resnet_type in ["RN50", "RN101", "RN152"]:
            output_shape = (2048, 7, 7)
        else:
            raise NotImplementedError(
                f"`RESNET_TYPE` must be one 'RNx' with x equaling one of"
                f" 18, 34, 50, 101, or 152."
            )

        rgb_sensor = next((s for s in self.sensors if isinstance(s, RGBSensor)), None)
        if rgb_sensor is not None:
            preprocessors.append(
                ResNetPreprocessor(
                    input_height=self.screen_size,
                    input_width=self.screen_size,
                    output_width=output_shape[2],
                    output_height=output_shape[1],
                    output_dims=output_shape[0],
                    pool=False,
                    torchvision_resnet_model=getattr(
                        models, f"resnet{self.resnet_type.replace('RN', '')}"
                    ),
                    input_uuids=[rgb_sensor.uuid],
                    output_uuid="rgb_resnet_imagenet",
                )
            )

        depth_sensor = next(
            (s for s in self.sensors if isinstance(s, DepthSensor)), None
        )
        if depth_sensor is not None:
            preprocessors.append(
                ResNetPreprocessor(
                    input_height=self.screen_size,
                    input_width=self.screen_size,
                    output_width=output_shape[2],
                    output_height=output_shape[1],
                    output_dims=output_shape[0],
                    pool=False,
                    torchvision_resnet_model=getattr(
                        models, f"resnet{self.resnet_type.replace('RN', '')}"
                    ),
                    input_uuids=[depth_sensor.uuid],
                    output_uuid="depth_resnet_imagenet",
                )
            )

        return preprocessors

    def create_model(self, **kwargs) -> nn.Module:
        has_rgb = any(isinstance(s, RGBSensor) for s in self.sensors)
        has_depth = any(isinstance(s, DepthSensor) for s in self.sensors)
        goal_sensor_uuid = next(
            (s.uuid for s in self.sensors if isinstance(s, self.goal_sensor_type)),
            None,
        )

        return ResnetTensorNavActorCritic(
            action_space=gym.spaces.Discrete(len(ObjectNavTask.class_action_names())),
            observation_space=kwargs["sensor_preprocessor_graph"].observation_spaces,
            goal_sensor_uuid=goal_sensor_uuid,
            rgb_resnet_preprocessor_uuid="rgb_resnet_imagenet" if has_rgb else None,
            depth_resnet_preprocessor_uuid="depth_resnet_imagenet"
            if has_depth
            else None,
            hidden_size=512,
            goal_dims=32,
        )


@attr.s(kw_only=True)
class ObjectNavUnfrozenResNetWithGRUActorCriticMixin:
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
            (s.uuid for s in self.sensors if isinstance(s, GoalObjectTypeThorSensor))
        )

        return ObjectNavActorCritic(
            action_space=gym.spaces.Discrete(len(ObjectNavTask.class_action_names())),
            observation_space=kwargs["sensor_preprocessor_graph"].observation_spaces,
            rgb_uuid=rgb_uuid,
            depth_uuid=depth_uuid,
            goal_sensor_uuid=goal_sensor_uuid,
            hidden_size=192
            if self.multiple_beliefs and len(self.auxiliary_uuids) > 1
            else 512,
            backbone=self.backbone,
            resnet_baseplanes=32,
            object_type_embedding_dim=32,
            num_rnn_layers=1,
            rnn_type="GRU",
            add_prev_actions=self.add_prev_actions,
            action_embed_size=6,
            auxiliary_uuids=self.auxiliary_uuids,
            multiple_beliefs=self.multiple_beliefs,
            beliefs_fusion=self.belief_fusion,
        )


class ObjectNavDAggerMixin:
    @staticmethod
    def training_pipeline(
        advance_scene_rollout_period: Optional[int] = None,
    ) -> TrainingPipeline:
        training_steps = int(300000000)
        tf_steps = int(5e6)
        anneal_steps = int(5e6)
        il_no_tf_steps = training_steps - tf_steps - anneal_steps
        assert il_no_tf_steps > 0

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
        return TrainingPipeline(
            save_interval=save_interval,
            metric_accumulate_interval=log_interval,
            optimizer_builder=Builder(optim.Adam, dict(lr=lr)),
            num_mini_batch=num_mini_batch,
            update_repeats=update_repeats,
            max_grad_norm=max_grad_norm,
            num_steps=num_steps,
            named_losses={"imitation_loss": Imitation(),},
            gamma=gamma,
            use_gae=use_gae,
            gae_lambda=gae_lambda,
            advance_scene_rollout_period=advance_scene_rollout_period,
            pipeline_stages=[
                PipelineStage(
                    loss_names=["imitation_loss"],
                    max_stage_steps=tf_steps,
                    teacher_forcing=LinearDecay(startp=1.0, endp=1.0, steps=tf_steps,),
                ),
                PipelineStage(
                    loss_names=["imitation_loss"],
                    max_stage_steps=anneal_steps + il_no_tf_steps,
                    teacher_forcing=LinearDecay(
                        startp=1.0, endp=0.0, steps=anneal_steps,
                    ),
                ),
            ],
            lr_scheduler_builder=Builder(
                LambdaLR, {"lr_lambda": LinearDecay(steps=training_steps)},
            ),
        )


def update_with_auxiliary_losses(
    named_losses: Dict[str, Tuple[AbstractActorCriticLoss, float]],
    auxiliary_uuids: Sequence[str],
    multiple_beliefs: bool,
) -> Dict[str, Tuple[AbstractActorCriticLoss, float]]:
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
    named_losses.update({uuid: total_aux_losses[uuid] for uuid in auxiliary_uuids})

    if multiple_beliefs:  # add weight entropy loss automatically
        named_losses[MultiAuxTaskNegEntropyLoss.UUID] = (
            MultiAuxTaskNegEntropyLoss(auxiliary_uuids),
            0.01,
        )

    return named_losses


class ObjectNavPPOMixin:
    @staticmethod
    def training_pipeline(
        auxiliary_uuids: Sequence[str],
        multiple_beliefs: bool,
        normalize_advantage: bool = True,
        advance_scene_rollout_period: Optional[int] = None,
        lr=3e-4,
        num_mini_batch=1,
        update_repeats=4,
        num_steps=128,
        save_interval=5000000,
        log_interval=10000 if torch.cuda.is_available() else 1,
        gamma=0.99,
        use_gae=True,
        gae_lambda=0.95,
        max_grad_norm=0.5,
    ) -> TrainingPipeline:
        ppo_steps = int(300000000)

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
