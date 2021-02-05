from projects.objectnav_baselines.experiments.robothor.objectnav_robothor_rgb_resnetgru_ddppo import (
    ObjectNaviThorRGBPPOExperimentConfig,
)


class ObjectNavRoboThorRGBPPOWSGINoObjFilterExperimentConfig(
    ObjectNaviThorRGBPPOExperimentConfig
):
    """An Object Navigation experiment configuration in RoboThor with RGB
    input using WSGI server."""

    USE_OBJECT_FILTER = False

    @classmethod
    def tag(cls):
        return "Objectnav-RoboTHOR-RGB-ResNetGRU-DDPPO-WSGI-NoObjFilter"

    def __init__(self):
        super().__init__()

        from ai2thor.wsgi_server import WsgiServer

        self.ENV_ARGS["server_class"] = WsgiServer

    def training_pipeline(self, **kwargs):
        pipeline = super(**kwargs)
        pipeline.save_interval = 1000000
        return pipeline
