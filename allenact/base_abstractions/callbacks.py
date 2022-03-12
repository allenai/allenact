class Callback:
    def setup(self, **kwargs) -> None:
        """Called once before training begins."""

    def on_train_log(self, **kwargs) -> None:
        """Called once train is supposed to log."""

    def on_valid_log(self, **kwargs) -> None:
        """Called after validation ends."""

    def on_test_log(self, **kwargs) -> None:
        """Called after test ends."""
