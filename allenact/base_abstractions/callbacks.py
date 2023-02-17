import abc
from typing import List, Dict, Any, Sequence, Optional

from allenact.base_abstractions.experiment_config import ExperimentConfig
from allenact.base_abstractions.sensor import Sensor

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


class Callback(abc.ABC):
    def setup(
        self,
        name: str,
        config: ExperimentConfig,
        mode: Literal["train", "valid", "test"],
        **kwargs,
    ) -> None:
        """Called once before training begins."""

    def on_train_log(
        self,
        *,
        metrics: List[Dict[str, Any]],
        metric_means: Dict[str, float],
        tasks_data: List[Any],
        step: int,
        scalar_name_to_total_experiences_key: Dict[str, str],
        **kwargs,
    ) -> None:
        """Called once train is supposed to log."""

    def on_valid_log(
        self,
        *,
        metrics: Dict[str, Any],
        metric_means: Dict[str, float],
        tasks_data: List[Any],
        step: int,
        scalar_name_to_total_experiences_key: Dict[str, str],
        checkpoint_file_name: str,
        **kwargs,
    ) -> None:
        """Called after validation ends."""

    def on_test_log(
        self,
        *,
        metrics: Dict[str, Any],
        metric_means: Dict[str, float],
        tasks_data: List[Any],
        step: int,
        scalar_name_to_total_experiences_key: Dict[str, str],
        checkpoint_file_name: str,
        **kwargs,
    ) -> None:
        """Called after test ends."""

    def after_save_project_state(self, base_dir: str) -> None:
        """Called after saving the project state in base_dir."""

    def callback_sensors(self) -> Optional[Sequence[Sensor]]:
        """Determines the data returned to the `tasks_data` parameter in the
        above *_log functions."""
