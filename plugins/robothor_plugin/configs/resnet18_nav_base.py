from typing import Dict, Any, List, Optional
import abc

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torchvision import models

from .nav_base import NavBaseConfig
from utils.experiment_utils import Builder, PipelineStage, TrainingPipeline, LinearDecay
from core.algorithms.onpolicy_sync.losses import PPO
from core.algorithms.onpolicy_sync.losses.ppo import PPOConfig
from core.base_abstractions.preprocessor import ObservationSet, Sensor
from plugins.habitat_plugin.habitat_preprocessors import ResnetPreProcessorHabitat
from utils.viz_utils import (
    SimpleViz,
    TrajectoryViz,
    ActorViz,
    AgentViewViz,
    TensorViz1D,
    TensorViz2D,
)
from plugins.robothor_plugin.robothor_viz import ThorViz


class Resnet18NavBaseConfig(NavBaseConfig, abc.ABC):
    """A Navigation base configuration in RoboTHOR."""

    RESNET_OUTPUT_UUID = "resnet"

    ENV_ARGS: Dict
    SCREEN_SIZE: int
    OBSERVATIONS: List[str]
    SENSORS: List[Sensor]

    def __init__(self):
        self.PREPROCESSORS = [
            Builder(
                ResnetPreProcessorHabitat,
                dict(
                    input_height=self.SCREEN_SIZE,
                    input_width=self.SCREEN_SIZE,
                    output_width=7,
                    output_height=7,
                    output_dims=512,
                    pool=False,
                    torchvision_resnet_model=models.resnet18,
                    input_uuids=[self.VISION_UUID],
                    output_uuid=self.RESNET_OUTPUT_UUID,
                    parallel=False,
                ),
            ),
        ]

    def machine_params(self, mode="train", **kwargs):
        res = super().machine_params(mode, **kwargs)

        nprocesses = res["nprocesses"]

        res["observation_set"] = (
            Builder(
                ObservationSet,
                kwargs=dict(
                    source_ids=self.OBSERVATIONS,
                    all_preprocessors=self.PREPROCESSORS,
                    all_sensors=self.SENSORS,
                ),
            )
            if (isinstance(nprocesses, int) and nprocesses > 0)
            or (isinstance(nprocesses, List) and max(nprocesses) > 0)
            else None
        )

        return res
