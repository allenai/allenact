from utils.experiment_utils import Builder
from projects.tutorials.pointnav_robothor_rgb_ddppo import (
    ObjectNavRoboThorRGBPPOExperimentConfig,
)
from utils.viz_utils import (
    SimpleViz,
    TrajectoryViz,
    ActorViz,
    AgentViewViz,
    TensorViz1D,
    TensorViz2D,
)
from plugins.robothor_plugin.robothor_viz import ThorViz


class ObjectNavRoboThorRGBPPOVizExperimentConfig(
    ObjectNavRoboThorRGBPPOExperimentConfig
):
    viz_ep_ids = [
        "FloorPlan_Train1_1_0",
        "FloorPlan_Train1_1_7",
        "FloorPlan_Train1_1_11",
        "FloorPlan_Train1_1_12",
    ]
    viz_video_ids = [["FloorPlan_Train1_1_7"], ["FloorPlan_Train1_1_11"]]

    def machine_params(self, mode="train", **kwargs):
        res = super().machine_params(mode, **kwargs)
        res["visualizer"] = None
        if mode == "test":
            res["visualizer"] = Builder(
                SimpleViz,
                dict(
                    episode_ids=self.viz_ep_ids,
                    mode=mode,
                    # Basic 2D trajectory visualizer (task output source):
                    v1=Builder(
                        TrajectoryViz,
                        dict(path_to_target_location=("task_info", "target",),),
                    ),
                    # Egocentric view visualizer (vector_task source):
                    v2=Builder(
                        AgentViewViz,
                        dict(max_video_length=100, episode_ids=self.viz_video_ids),
                    ),
                    # Default action probability visualizer (actor critic output source):
                    v3=Builder(ActorViz, dict(figsize=(3.25, 10), fontsize=18)),
                    # Default taken action logprob visualizer (rollout storage source):
                    v4=Builder(TensorViz1D, dict()),
                    # Same episode mask visualizer (rollout storage source):
                    v5=Builder(TensorViz1D, dict(rollout_source=("masks"))),
                    # Default recurrent memory visualizer (rollout storage source):
                    v6=Builder(TensorViz2D, dict()),
                    # Specialized 2D trajectory visualizer (task output source):
                    v7=Builder(
                        ThorViz,
                        dict(
                            figsize=(16, 8),
                            viz_rows_cols=(448, 448),
                            scenes=("FloorPlan_Train{}_{}", 1, 1, 1, 1),
                        ),
                    ),
                ),
            )

        return res
