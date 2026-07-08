"""Early stopping utility to halt training when the validation metric stalls."""

import logging
from src.utils.logger import setup_logger
logger: logging.Logger = setup_logger("early_stopping")


class EarlyStopping:
    """Tracks validation metric history and signals when to stop training."""

    def __init__(self, patience: int, mode: str = "max") -> None:
        """Initializes the early stopping tracker.

        Args:
            patience: Number of epochs with no improvement before stopping.
            mode: "max" if a higher metric is better, "min" otherwise.
        """
        self.patience = patience
        self.mode = mode
        self.best_value: float = float("-inf") if mode == "max" else float("inf")
        self.epochs_without_improvement: int = 0
        self.should_stop: bool = False

    def step(self, current_value: float) -> bool:
        """Updates the tracker with the latest metric value.

        Args:
            current_value: Validation metric value for the current epoch.

        Returns:
            True if this is the best value seen so far, else False.
        """
        is_improvement = (
            current_value > self.best_value if self.mode == "max" else current_value < self.best_value
        )
        if is_improvement:
            self.best_value = current_value
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1

        if self.epochs_without_improvement >= self.patience:
            self.should_stop = True
            logger.info("Early stopping triggered: no improvement for %d epochs.", self.patience)
        return is_improvement