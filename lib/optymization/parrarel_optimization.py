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
    min_percentage_improvement: float,
    output_dir_path: Path,
    hyper_opt_prefix: str,
    study_prefix: str,
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

    # --- TODO ---
    # 1. Differentiate between the models that by default can use multiple cores and those that can't.
    # 2. If the model can't use multiple cores, then set mutliple models to run at the same time.
    # 3. If the model can use multiple cores, then set only one model to run at the same time.

    parallel_model_prefixes = ["Ridge", "KNeighbors"]
    sequential_model_prefixes = ["RandomForest", "XGB", "LGBM"]

    sequenctial_model_combinations = [
        model_combination
        for model_combination in all_model_combinations
        if any(  # type: ignore
            model_combination.name.startswith(prefix)
            for prefix in sequential_model_prefixes
        )
    ]

    parallel_model_combinations = [
        model_combination
        for model_combination in all_model_combinations
        if any(  # type: ignore
            model_combination.name.startswith(prefix)
            for prefix in parallel_model_prefixes
        )
    ]
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
                    min_percentage_improvement,
                    i,
                    output_dir_path,
                    hyper_opt_prefix,
                    study_prefix,
                    create_objective,
                )
                for i, model_combination in enumerate(parallel_model_combinations)
                if model_combination.name not in omit_names
            ],
        )

    with mp.Pool(processes=1, initializer=init_worker) as pool:
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
                    min_percentage_improvement,
                    i,
                    output_dir_path,
                    hyper_opt_prefix,
                    study_prefix,
                    create_objective,
                )
                for i, model_combination in enumerate(sequenctial_model_combinations)
                if model_combination.name not in omit_names
            ],
        )
