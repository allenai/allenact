from typing import Optional

from plugins.robothor_plugin.robothor_viz import ThorViz
from projects.tutorials.pointnav_robothor_rgb_ddppo import (
    PointNavRoboThorRGBPPOExperimentConfig,
)
from utils.viz_utils import (
    VizSuite,
    TrajectoryViz,
    ActorViz,
    AgentViewViz,
    TensorViz1D,
    TensorViz2D,
)


class PointNavRoboThorRGBPPOVizExperimentConfig(PointNavRoboThorRGBPPOExperimentConfig):
    """ExperimentConfig used to demonstrate how to set up visualization code.

    # Attributes

    viz_ep_ids : Scene names that will be visualized.
    viz_video_ids : Scene names that will have videos visualizations associated with them.
    """

    viz_ep_ids = [
        "FloorPlan_Train1_1_3",
        "FloorPlan_Train1_1_4",
        "FloorPlan_Train1_1_5",
        "FloorPlan_Train1_1_6",
    ]
    viz_video_ids = [["FloorPlan_Train1_1_3"], ["FloorPlan_Train1_1_4"]]

    viz: Optional[VizSuite] = None

    def get_viz(self, mode):
        if self.viz is not None:
            return self.viz

        self.viz = VizSuite(
            episode_ids=self.viz_ep_ids,
            mode=mode,
            # Basic 2D trajectory visualizer (task output source):
            base_trajectory=TrajectoryViz(
                path_to_target_location=("task_info", "target",),
            ),
            # Egocentric view visualizer (vector task source):
            egeocentric=AgentViewViz(
                max_video_length=100, episode_ids=self.viz_video_ids
            ),
            # Default action probability visualizer (actor critic output source):
            action_probs=ActorViz(figsize=(3.25, 10), fontsize=18),
            # Default taken action logprob visualizer (rollout storage source):
            taken_action_logprobs=TensorViz1D(),
            # Same episode mask visualizer (rollout storage source):
            episode_mask=TensorViz1D(rollout_source=("masks",)),
            # Default recurrent memory visualizer (rollout storage source):
            rnn_memory=TensorViz2D(),
            # Specialized 2D trajectory visualizer (task output source):
            thor_trajectory=ThorViz(
                figsize=(16, 8),
                viz_rows_cols=(448, 448),
                scenes=("FloorPlan_Train{}_{}", 1, 1, 1, 1),
            ),
        )

        return self.viz

    def machine_params(self, mode="train", **kwargs):
        res = super().machine_params(mode, **kwargs)
        res["visualizer"] = None
        if mode == "test":
            res["visualizer"] = self.get_viz(mode)

        return res
