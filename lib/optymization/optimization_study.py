import os
from pathlib import Path
from typing import Callable
import gc

import optuna
import pandas as pd
import pickle

from lib.models.HyperOptCombination import HyperOptCombination
from lib.models.HyperOptResultDict import HyperOptResultDict
from lib.optymization.EarlyStoppingCallback import EarlyStoppingCallback
from lib.logger import setup_logger


logger = setup_logger(__name__)


def optimize_model_and_save(
    model_run: str,
    direction: str,
    model_combination: HyperOptCombination,
    X: pd.DataFrame,
    y: pd.DataFrame | pd.Series,
    n_optimization_trials: int,
    n_cv: int,
    n_patience: int,
    min_percentage_improvement: float,
    i: int,
    output_dir_path: Path,
    hyper_opt_prefix: str,
    study_prefix: str,
    create_objective_func: Callable[
        [pd.DataFrame, pd.DataFrame | pd.Series, HyperOptCombination, int],
        Callable[[optuna.Trial], float],
    ],
) -> None:
    combination_name = model_combination.name
    logger.info(f"Optimizing model combination {i}: {combination_name}")

    X = X.copy()[model_combination.feature_combination.features]
    y = y.copy()

    early_stopping = EarlyStoppingCallback(
        name=model_combination.name,
        patience=n_patience,
        min_percentage_improvement=min_percentage_improvement,
    )

    try:
        os.makedirs(os.path.join(output_dir_path, f"{study_prefix}{model_run}"))
    except OSError:
        pass

    # Create an Optuna study for hyperparameter optimization
    study = optuna.create_study(
        direction=direction,
        study_name=f"optuna_{model_combination.name}",
        load_if_exists=True,
        storage=f"sqlite:///{output_dir_path}/{study_prefix}{model_run}/{model_combination.name}.db",
    )

    study.optimize(
        func=create_objective_func(X, y, model_combination, n_cv),
        n_trials=n_optimization_trials,
        callbacks=[early_stopping],  # type: ignore
    )

    # Save the best parameters and score
    best_params = study.best_params
    best_score = study.best_value

    logger.info(
        f"Model combination: {combination_name} has scored: {best_score} with params: {best_params}"
    )

    result = HyperOptResultDict(
        name=model_combination.name,
        score=best_score,
        params=best_params,
        model=model_combination.model,
        features=model_combination.feature_combination.features,
    )

    try:
        os.makedirs(os.path.join(output_dir_path, f"{hyper_opt_prefix}{model_run}"))
    except OSError:
        pass

    try:
        results_path = os.path.join(
            output_dir_path,
            f"{hyper_opt_prefix}{model_run}",
            f"{model_combination.name}.pkl",
        )
        pickle.dump(result, open(results_path, "wb"))
    except Exception as e:
        logger.error(f"Error saving model combination {combination_name}: {e}")
        raise e

    del X
    del y
    del study
    gc.collect()
