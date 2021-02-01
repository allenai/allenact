from projects.objectnav_baselines.experiments.robothor.objectnav_robothor_rgb_resnetgru_ddppo import (
    ObjectNaviThorRGBPPOExperimentConfig,
)


class ObjectNavRoboThorRGBPPOWSGIExperimentConfig(ObjectNaviThorRGBPPOExperimentConfig):
    """An Object Navigation experiment configuration in RoboThor with RGB
    input using WSGI server."""

    @classmethod
    def tag(cls):
        return "Objectnav-RoboTHOR-RGB-ResNetGRU-DDPPO-WSGI"

    def __init__(self):
        super().__init__()

        from ai2thor.wsgi_server import WsgiServer

        self.ENV_ARGS["server_class"] = WsgiServer
