from pathlib import Path
import signal
from typing import Callable, List, Optional
import multiprocessing as mp

import optuna
import pandas as pd

from lib.models.HyperOptCombination import HyperOptCombination
from lib.optymization.optimization_study import optimize_model_and_save
from lib.logger import setup_logger

logger = setup_logger(__name__)


def init_worker() -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def run_parallel_optimization(
    model_run: str,
    direction: str,
    all_model_combinations: List[HyperOptCombination],
    X: pd.DataFrame,
    y: pd.DataFrame | pd.Series,
    n_optimization_trials: int,
    n_cv: int,
    n_patience: int,
    output_dir_path: Path,
    hyper_opt_prefix: str,
    create_objective: Callable[
        [pd.DataFrame, pd.DataFrame | pd.Series, HyperOptCombination, int],
        Callable[[optuna.Trial], float],
    ],
    omit_names: List[str] = [],
    processes: Optional[int] = None,
) -> None:
    if processes is None:
        processes = mp.cpu_count() * 3 // 4

    logger.info(f"Running optimization with {processes} processes")

    # Set up multiprocessing pool
    with mp.Pool(processes=processes, initializer=init_worker) as pool:
        # Map each iteration of the loop to a process
        _ = pool.starmap(
            optimize_model_and_save,
            [
                (
                    model_run,
                    direction,
                    model_combination,
                    X,
                    y,
                    n_optimization_trials,
                    n_cv,
                    n_patience,
                    i,
                    output_dir_path,
                    hyper_opt_prefix,
                    create_objective,
                )
                for i, model_combination in enumerate(all_model_combinations)
                if model_combination.name not in omit_names
            ],
        )
