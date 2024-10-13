import json
from pathlib import Path
import signal
from typing import Callable, Dict, List, Optional
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
    metadata: Optional[Dict] = None,
) -> None:
    if processes is None:
        processes = -1
        # processes = mp.cpu_count() * 3 // 4

    logger.info(f"Running optimization with {processes} processes")

    # --- TODO ---
    # 1. Differentiate between the models that by default can use multiple cores and those that can't.
    # 2. If the model can't use multiple cores, then set mutliple models to run at the same time.
    # 3. If the model can use multiple cores, then set only one model to run at the same time.

    parallel_model_prefixes = ["ridge"]
    omit_mulit_sufixes = ["top_0", "top_1", "top_2"]

    sequential_model_combinations = [
        model_combination
        for model_combination in all_model_combinations
        if all(  # type: ignore
            not model_combination.name.lower().startswith(prefix.lower())
            for prefix in parallel_model_prefixes
        )
        and all(
            f"{model_combination.name}{omit_sufix}" not in omit_names
            for omit_sufix in omit_mulit_sufixes
        )
    ]

    parallel_model_combinations = [
        model_combination
        for model_combination in all_model_combinations
        if any(  # type: ignore
            model_combination.name.lower().startswith(prefix.lower())
            for prefix in parallel_model_prefixes
        )
        and all(
            f"{model_combination.name}{omit_sufix}" not in omit_names
            for omit_sufix in omit_mulit_sufixes
        )
    ]
    logger.info(
        "Will be running parallel optimization for models: "
        + json.dumps([model.name for model in parallel_model_combinations], indent=4)
    )
    logger.info(
        "Will be running sequential optimization for models: "
        + json.dumps([model.name for model in sequential_model_combinations], indent=4)
    )
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
                    metadata,
                )
                for i, model_combination in enumerate(parallel_model_combinations)
                if model_combination.name not in omit_names
            ],
        )

    for i, model_combination in enumerate(sequential_model_combinations):
        if model_combination.name in omit_names:
            continue

        optimize_model_and_save(
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
            metadata,
        )
