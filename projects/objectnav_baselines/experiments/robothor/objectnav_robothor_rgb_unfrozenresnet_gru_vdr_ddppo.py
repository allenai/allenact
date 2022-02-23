from typing import Sequence, Union, Optional, Any

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

from allenact.algorithms.onpolicy_sync.losses import PPO
from allenact.algorithms.onpolicy_sync.losses.ppo import PPOConfig
from allenact.algorithms.onpolicy_sync.storage import RolloutBlockStorage
from allenact.base_abstractions.preprocessor import Preprocessor

# noinspection PyUnresolvedReferences
from allenact.base_abstractions.sensor import Sensor
from allenact.base_abstractions.task import Task
from allenact.embodiedai.sensors.vision_sensors import RGBSensor, DepthSensor
from allenact.embodiedai.storage.vdr_storage import (
    DiscreteVisualDynamicsReplayStorage,
    InverseDynamicsVDRLoss,
)
from allenact.utils.experiment_utils import Builder, TrainingSettings
from allenact.utils.experiment_utils import (
    PipelineStage,
    TrainingPipeline,
    LinearDecay,
    StageComponent,
)
from allenact_plugins.ithor_plugin.ithor_environment import IThorEnvironment
from allenact_plugins.ithor_plugin.ithor_sensors import GoalObjectTypeThorSensor
from allenact_plugins.ithor_plugin.ithor_sensors import RGBSensorThor
from allenact_plugins.robothor_plugin.robothor_environment import RoboThorEnvironment
from allenact_plugins.robothor_plugin.robothor_tasks import ObjectNavTask
from projects.objectnav_baselines.experiments.objectnav_mixin_ddppo import (
    ObjectNavMixInPPOConfig,
)
from projects.objectnav_baselines.experiments.robothor.objectnav_robothor_base import (
    ObjectNavRoboThorBaseConfig,
)
from projects.objectnav_baselines.models.object_nav_models import ObjectNavActorCritic


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


class ObjectNavRoboThorVdrTmpRGBExperimentConfig(
    ObjectNavRoboThorBaseConfig, ObjectNavMixInPPOConfig,
):
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

    @classmethod
    def tag(cls):
        return "Objectnav-RoboTHOR-RGB-Unfrozen-VDR"

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

    @classmethod
    def preprocessors(cls) -> Sequence[Union[Preprocessor, Builder[Preprocessor]]]:
        return []

    BACKBONE = "gnresnet18"

    @classmethod
    def create_model(cls, **kwargs) -> nn.Module:
        rgb_uuid = next((s.uuid for s in cls.SENSORS if isinstance(s, RGBSensor)), None)
        depth_uuid = next(
            (s.uuid for s in cls.SENSORS if isinstance(s, DepthSensor)), None
        )
        goal_sensor_uuid = next(
            (s.uuid for s in cls.SENSORS if isinstance(s, GoalObjectTypeThorSensor)),
            None,
        )

        model = ObjectNavActorCritic(
            action_space=gym.spaces.Discrete(len(ObjectNavTask.class_action_names())),
            observation_space=kwargs["sensor_preprocessor_graph"].observation_spaces,
            rgb_uuid=rgb_uuid,
            depth_uuid=depth_uuid,
            goal_sensor_uuid=goal_sensor_uuid,
            hidden_size=192
            if cls.MULTIPLE_BELIEFS and len(cls.AUXILIARY_UUIDS) > 1
            else 512,
            backbone=cls.BACKBONE,
            resnet_baseplanes=32,
            object_type_embedding_dim=32,
            num_rnn_layers=1,
            rnn_type="GRU",
            add_prev_actions=cls.ADD_PREV_ACTIONS,
            action_embed_size=6,
            auxiliary_uuids=cls.AUXILIARY_UUIDS,
            multiple_beliefs=cls.MULTIPLE_BELIEFS,
            beliefs_fusion=cls.BELIEF_FUSION,
        )
        model.inv_dyn_mlp = nn.Sequential(
            nn.Linear(1024, 256), nn.ReLU(inplace=True), nn.Linear(256, 6),
        )

        return model
