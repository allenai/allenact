from typing import Union, Optional, Any

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

from allenact.algorithms.onpolicy_sync.losses import PPO
from allenact.algorithms.onpolicy_sync.losses.ppo import PPOConfig
from allenact.algorithms.onpolicy_sync.storage import RolloutBlockStorage

# noinspection PyUnresolvedReferences
from allenact.base_abstractions.sensor import Sensor
from allenact.base_abstractions.task import Task
from allenact.embodiedai.storage.vdr_storage import (
    DiscreteVisualDynamicsReplayStorage,
    InverseDynamicsVDRLoss,
)
from allenact.utils.experiment_utils import Builder, TrainingSettings
from allenact.utils.experiment_utils import (
    PipelineStage,
    LinearDecay,
    StageComponent,
)
from allenact.utils.experiment_utils import TrainingPipeline
from allenact_plugins.ithor_plugin.ithor_environment import IThorEnvironment
from allenact_plugins.ithor_plugin.ithor_sensors import (
    RGBSensorThor,
    GoalObjectTypeThorSensor,
)
from allenact_plugins.robothor_plugin.robothor_environment import RoboThorEnvironment
from projects.objectnav_baselines.experiments.robothor.objectnav_robothor_base import (
    ObjectNavRoboThorBaseConfig,
)
from projects.objectnav_baselines.mixins import (
    ObjectNavUnfrozenResNetWithGRUActorCriticMixin,
    update_with_auxiliary_losses,
)


def compute_inv_dyn_action_logits(
    model, img0, img1,
):
    rgb_uuid = model.visual_encoder.rgb_uuid
    img0_enc = model.visual_encoder({rgb_uuid: img0.unsqueeze(0)}).squeeze(0)
    img1_enc = model.visual_encoder({rgb_uuid: img1.unsqueeze(0)}).squeeze(0)
    return model.inv_dyn_mlp(torch.cat((img0_enc, img1_enc), dim=1))


class LastActionSuccessSensor(
    Sensor[
        Union[IThorEnvironment, RoboThorEnvironment],
        Union[Task[IThorEnvironment], Task[RoboThorEnvironment]],
    ]
):
    def __init__(self, uuid: str = "last_action_success", **kwargs: Any):
        super().__init__(
            uuid=uuid, observation_space=gym.spaces.MultiBinary(1), **kwargs
        )

    def get_observation(
        self,
        env: Union[IThorEnvironment, RoboThorEnvironment],
        task: Optional[Task],
        *args: Any,
        **kwargs: Any
    ) -> Any:
        return 1 * task.last_action_success


class VisibleObjectTypesSensor(
    Sensor[
        Union[IThorEnvironment, RoboThorEnvironment],
        Union[Task[IThorEnvironment], Task[RoboThorEnvironment]],
    ]
):
    def __init__(self, uuid: str = "visible_objects", **kwargs: Any):
        super().__init__(
            uuid=uuid,
            observation_space=gym.spaces.Box(
                low=0, high=1, shape=(len(ObjectNavRoboThorBaseConfig.TARGET_TYPES),)
            ),
            **kwargs
        )
        self.type_to_index = {
            tt: i for i, tt in enumerate(ObjectNavRoboThorBaseConfig.TARGET_TYPES)
        }

    def get_observation(
        self,
        env: Union[IThorEnvironment, RoboThorEnvironment],
        task: Optional[Task],
        *args: Any,
        **kwargs: Any
    ) -> Any:
        out = np.zeros((len(self.type_to_index),))
        for o in env.controller.last_event.metadata["objects"]:
            if o["visible"] and o["objectType"] in self.type_to_index:
                out[self.type_to_index[o["objectType"]]] = 1.0
        return out


class ObjectNavRoboThorVdrTmpRGBExperimentConfig(ObjectNavRoboThorBaseConfig):
    SENSORS = [
        RGBSensorThor(
            height=ObjectNavRoboThorBaseConfig.SCREEN_SIZE,
            width=ObjectNavRoboThorBaseConfig.SCREEN_SIZE,
            use_resnet_normalization=True,
            uuid="rgb_lowres",
        ),
        GoalObjectTypeThorSensor(
            object_types=ObjectNavRoboThorBaseConfig.TARGET_TYPES,
        ),
        LastActionSuccessSensor(),
        VisibleObjectTypesSensor(),
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.model_creation_handler = ObjectNavUnfrozenResNetWithGRUActorCriticMixin(
            backbone="gnresnet18",
            sensors=self.SENSORS,
            auxiliary_uuids=[],
            add_prev_actions=True,
            multiple_beliefs=False,
            belief_fusion=None,
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
        named_losses = update_with_auxiliary_losses(named_losses)

        default_ts = TrainingSettings(
            num_mini_batch=num_mini_batch,
            update_repeats=update_repeats,
            max_grad_norm=max_grad_norm,
            num_steps=num_steps,
            gamma=gamma,
            use_gae=use_gae,
            gae_lambda=gae_lambda,
            advance_scene_rollout_period=self.ADVANCE_SCENE_ROLLOUT_PERIOD,
            save_interval=save_interval,
            metric_accumulate_interval=log_interval,
        )

        named_losses = {
            **named_losses,
            "inv_dyn_vdr": (
                InverseDynamicsVDRLoss(
                    compute_action_logits_fn=compute_inv_dyn_action_logits,
                    img0_key="img0",
                    img1_key="img1",
                    action_key="action",
                ),
                1.0,
            ),
        }

        sorted_loss_names = list(sorted(named_losses.keys()))
        return TrainingPipeline(
            training_settings=default_ts,
            optimizer_builder=Builder(optim.Adam, dict(lr=lr)),
            named_losses={k: v[0] for k, v in named_losses.items()},
            named_storages={
                "onpolicy": RolloutBlockStorage(init_size=num_steps),
                "discrete_vdr": DiscreteVisualDynamicsReplayStorage(
                    image_uuid="rgb_lowres",
                    action_success_uuid="last_action_success",
                    extra_targets=["visible_objects"],
                    nactions=6,
                    num_to_store_per_action=200 if torch.cuda.is_available() else 10,
                    max_to_save_per_episode=6,
                    target_batch_size=256 if torch.cuda.is_available() else 128,
                ),
            },
            pipeline_stages=[
                PipelineStage(
                    loss_names=sorted_loss_names,
                    max_stage_steps=ppo_steps,
                    loss_weights=[
                        named_losses[loss_name][1] for loss_name in sorted_loss_names
                    ],
                    stage_components=[
                        StageComponent(
                            uuid="onpolicy",
                            storage_uuid="onpolicy",
                            loss_names=[
                                ln for ln in sorted_loss_names if ln != "inv_dyn_vdr"
                            ],
                        ),
                        StageComponent(
                            uuid="vdr",
                            storage_uuid="discrete_vdr",
                            loss_names=["inv_dyn_vdr"],
                            training_settings=TrainingSettings(
                                num_mini_batch=1, update_repeats=1,
                            ),
                        ),
                    ],
                )
            ],
            lr_scheduler_builder=Builder(
                LambdaLR, {"lr_lambda": LinearDecay(steps=ppo_steps)}
            ),
        )

    def create_model(self, **kwargs) -> nn.Module:
        model = self.model_creation_handler.create_model(**kwargs)
        model.inv_dyn_mlp = nn.Sequential(
            nn.Linear(1024, 256), nn.ReLU(inplace=True), nn.Linear(256, 6),
        )
        return model

    def tag(self):
        return "Objectnav-RoboTHOR-RGB-UnfrozenResNet18GRU-VDR"
