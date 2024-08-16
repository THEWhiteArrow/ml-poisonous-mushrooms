import signal
from typing import Callable, List
import multiprocessing as mp

import optuna
import pandas as pd

from lib.models.HyperOptCombination import HyperOptCombination
from lib.optymization.optimization_study import optimize_model_and_save


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
    model_dir_path: str,
    create_objective: Callable[
        [pd.DataFrame, pd.DataFrame | pd.Series, HyperOptCombination, int],
        Callable[[optuna.Trial], float],
    ],
    omit_names: List[str] = [],
) -> None:

    # Set up multiprocessing pool
    with mp.Pool(processes=mp.cpu_count() * 3 // 4, initializer=init_worker) as pool:
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
                    model_dir_path,
                    create_objective,
                )
                for i, model_combination in enumerate(all_model_combinations)
                if model_combination.name not in omit_names
            ],
        )
