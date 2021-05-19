import os
from abc import ABC

import torch

from projects.objectnav_baselines.experiments.objectnav_thor_base import (
    ObjectNavThorBaseConfig,
)


class ObjectNaviThorBaseConfig(ObjectNavThorBaseConfig, ABC):
    """The base config for all iTHOR ObjectNav experiments.

    Only working in Kitchen scenes at the moment.
    """

    DEFAULT_NUM_TRAIN_PROCESSES = 40 if torch.cuda.is_available() else 1

    TRAIN_DATASET_DIR = os.path.join(os.getcwd(), "datasets/ithor-objectnav/train")
    VAL_DATASET_DIR = os.path.join(os.getcwd(), "datasets/ithor-objectnav/val")

    AGENT_MODE: str = "default"

    TARGET_TYPES = tuple(
        sorted(
            [
                "Apple",
                "Bowl",
                "Bread",
                "ButterKnife",
                "CoffeeMachine",
                "Cup",
                "DishSponge",
                "Egg",
                "Faucet",
                "Fork",
                "Fridge",
                "Kettle",
                "Knife",
                "Ladle",
                "Lettuce",
                "LightSwitch",
                "Microwave",
                "Mug",
                "Pan",
                "PepperShaker",
                "Plate",
                "Pot",
                "Potato",
                "SaltShaker",
                "Sink",
                "SoapBottle",
                "Spatula",
                "Spoon",
                "StoveBurner",
                "Toaster",
                "Tomato",
                "WineBottle"
            ],
        )
    )
