class Callback:
    def setup(self, **kwargs) -> None:
        """Called once before training begins."""

    def on_train_log(self, **kwargs) -> None:
        """Called once train is supposed to log."""

    def on_valid_log(self, **kwargs) -> None:
        """Called after validation ends."""

    def on_test_log(self, **kwargs) -> None:
        """Called after test ends."""

    def after_save_project_state(self, base_dir: str) -> None:
        """Called after saving the project state in base_dir."""
