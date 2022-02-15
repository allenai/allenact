from projects.objectnav_baselines.experiments.objectnav_mixin_resnetgru import (
    ObjectNavMixInResNetGRUConfig,
)


class ObjectNavMixInResNet18GRUConfig(ObjectNavMixInResNetGRUConfig):
    RESNET_TYPE: str = "RN18"
