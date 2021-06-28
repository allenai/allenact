from abc import ABC

from allenact_plugins.manipulathor_plugin.armpointnav_constants import TRAIN_OBJECTS, TEST_OBJECTS
from projects.manipulathor_baselines.armpointnav_baselines.experiments.armpointnav_thor_base import ArmPointNavThorBaseConfig


class ArmPointNaviThorBaseConfig(ArmPointNavThorBaseConfig, ABC):
    """The base config for all iTHOR ObjectNav experiments."""

    NUM_PROCESSES = 40
    # add all the arguments here
    TOTAL_NUMBER_SCENES = 30

    TRAIN_SCENES = [
        "FloorPlan{}_physics".format(str(i))
        for i in range(1, TOTAL_NUMBER_SCENES + 1)
        if (i % 3 == 1 or i % 3 == 0) and i != 28
    ]  # last scenes are really bad
    TEST_SCENES = [
        "FloorPlan{}_physics".format(str(i))
        for i in range(1, TOTAL_NUMBER_SCENES + 1)
        if i % 3 == 2 and i % 6 == 2
    ]
    VALID_SCENES = [
        "FloorPlan{}_physics".format(str(i))
        for i in range(1, TOTAL_NUMBER_SCENES + 1)
        if i % 3 == 2 and i % 6 == 5
    ]

    ALL_SCENES = TRAIN_SCENES + TEST_SCENES + VALID_SCENES

    assert (
        len(ALL_SCENES) == TOTAL_NUMBER_SCENES - 1
        and len(set(ALL_SCENES)) == TOTAL_NUMBER_SCENES - 1
    )

    OBJECT_TYPES = tuple(sorted(TRAIN_OBJECTS))

    UNSEEN_OBJECT_TYPES = tuple(sorted(TEST_OBJECTS))
