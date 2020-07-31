import os
from pathlib import Path

BABYAI_EXPERT_TRAJECTORIES_DIR = os.path.abspath(
    os.path.join(os.path.dirname(Path(__file__)), "babyai_data", "demos")
)
