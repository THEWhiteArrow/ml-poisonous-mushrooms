from dataclasses import dataclass
from typing import Optional

import optuna

from lib.logger import setup_logger


logger = setup_logger(__name__)


@dataclass
class EarlyStoppingCallback:
    name: str
    patience: int
    best_value: Optional[float] = None
    no_improvement_count: int = 0

    def __call__(self, study: optuna.Study, trial: optuna.Trial):
        # Get the current best value
        current_best_value = study.best_value

        # Check if the best value has improved
        if self.best_value is None or current_best_value > self.best_value:
            self.best_value = current_best_value
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1

        # Stop study if there has been no improvement for `self.patience` trials
        if self.no_improvement_count >= self.patience:
            logger.info(
                f"Early stopping the study: {
                    self.name} due to no improvement for {self.patience} trials"
            )
            study.stop()
