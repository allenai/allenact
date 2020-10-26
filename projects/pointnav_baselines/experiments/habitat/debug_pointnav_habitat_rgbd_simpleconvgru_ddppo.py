import gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

from core.algorithms.onpolicy_sync.losses import PPO
from core.algorithms.onpolicy_sync.losses.ppo import PPOConfig
from plugins.habitat_plugin.habitat_sensors import (
    RGBSensorHabitat,
    TargetCoordinatesSensorHabitat, DepthSensorHabitat,
)
from plugins.habitat_plugin.habitat_tasks import PointNavTask
from plugins.habitat_plugin.habitat_utils import construct_env_configs
from projects.pointnav_baselines.experiments.habitat.debug_pointnav_habitat_base import DebugPointNavHabitatBaseConfig
from projects.pointnav_baselines.models.point_nav_models import (
    PointNavActorCriticSimpleConvRNN,
)
from utils.experiment_utils import Builder, PipelineStage, TrainingPipeline, LinearDecay


class PointNavHabitatRGBDDeterministiSimpleConvGRUDDPPOExperimentConfig(
    DebugPointNavHabitatBaseConfig
):
    """An Point Navigation experiment configuration in Habitat with Depth
    input."""

    def __init__(self):
        super().__init__()
        self.SENSORS = [
            RGBSensorHabitat(
                height=self.SCREEN_SIZE,
                width=self.SCREEN_SIZE,
                use_resnet_normalization=True,
            ),
            DepthSensorHabitat(
                height=self.SCREEN_SIZE,
                width=self.SCREEN_SIZE,
                use_normalization=True,
            ),
            TargetCoordinatesSensorHabitat(coordinate_dims=2),
        ]

        self.PREPROCESSORS = []

        self.OBSERVATIONS = [
            "rgb",
            "depth",
            "target_coordinates_ind",
        ]

        self.CONFIG = self.CONFIG.clone()
        self.CONFIG.SIMULATOR.AGENT_0.SENSORS = ["DEPTH_SENSOR", "RGB_SENSOR"]

        self.TRAIN_CONFIGS = construct_env_configs(config=self.CONFIG, allow_scene_repeat=True)

    @classmethod
    def tag(cls):
        return "Debug-Pointnav-Habitat-RGBD-SimpleConv-DDPPO"

    @classmethod
    def training_pipeline(cls, **kwargs):
        ppo_steps = int(75000000)
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
            named_losses={"ppo_loss": PPO(**PPOConfig)},
            gamma=gamma,
            use_gae=use_gae,
            gae_lambda=gae_lambda,
            advance_scene_rollout_period=cls.ADVANCE_SCENE_ROLLOUT_PERIOD,
            pipeline_stages=[
                PipelineStage(loss_names=["ppo_loss"], max_stage_steps=ppo_steps)
            ],
            lr_scheduler_builder=Builder(
                LambdaLR, {"lr_lambda": LinearDecay(steps=ppo_steps)}
            ),
        )

    @classmethod
    def create_model(cls, **kwargs) -> nn.Module:
        return PointNavActorCriticSimpleConvRNN(
            action_space=gym.spaces.Discrete(len(PointNavTask.class_action_names())),
            observation_space=kwargs["observation_set"].observation_spaces,
            rgb_uuid="rgb",
            depth_uuid="depth",
            goal_sensor_uuid="target_coordinates_ind",
            hidden_size=512,
            embed_coordinates=False,
            coordinate_embedding_dim=2,
            coordinate_dims=2,
            num_rnn_layers=1,
            rnn_type="GRU",
        )
