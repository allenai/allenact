from abc import ABC
from typing import Dict, Any

from allenact.utils.viz_utils import VizSuite, AgentViewViz

from projects.gym_baselines.experiments.gym_base import GymBaseConfig


class GymMoJoCoBaseConfig(GymBaseConfig, ABC):
    @classmethod
    def machine_params(cls, mode="train", **kwargs) -> Dict[str, Any]:
        visualizer = None
        if mode == "test":
            visualizer = VizSuite(
                mode=mode,
                video_viz=AgentViewViz(
                    label="episode_vid",
                    max_clip_length=400,
                    vector_task_source=("render", {"mode": "rgb_array"}),
                    fps=30,
                ),
            )
        return {
            "nprocesses": 8 if mode == "train" else 1,  # rollout
            "devices": [],
            "visualizer": visualizer,
        }
