
import os
from typing import Callable

import optuna
import pandas as pd

from lib.models.HyperOptCombination import HyperOptCombination
from lib.models.HyperOptResult import HyperOptResult
from lib.optymization.EarlyStoppingCallback import EarlyStoppingCallback
from lib.logger import setup_logger
from models import HYPER_OPT_PREFIX


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
    i: int,
    model_dir_path: str,
    create_objective_func: Callable[[pd.DataFrame, pd.DataFrame | pd.Series, HyperOptCombination, int], Callable[[optuna.Trial], float]],
) -> None:
    logger.info(f"Optimizing model combination {i}: {model_combination.name}")

    X = X.copy()[model_combination.feature_combination.features]
    y = y.copy()

    early_stopping = EarlyStoppingCallback(name=model_combination.name, patience=n_patience)

    # Create an Optuna study for hyperparameter optimization
    study = optuna.create_study(direction=direction, study_name=f"optuna_{model_combination.name}")

    study.optimize(
        func=create_objective_func(X, y, model_combination, n_cv),
        n_trials=n_optimization_trials,
        callbacks=[early_stopping],  # type: ignore
    )

    # Save the best parameters and score
    best_params = study.best_params
    best_score = study.best_value

    logger.info(
        f"Model combination: {model_combination.name} has scored: {
            best_score} with params: {best_params}"
    )

    result = HyperOptResult(
        name=model_combination.name,
        score=best_score,
        params=best_params,
        model=model_combination.model_wrapper.model,
        features=model_combination.feature_combination.features,
    )

    result.pickle(
        path=os.path.join(
            model_dir_path,
            f"{HYPER_OPT_PREFIX}{model_run}",
            f"{model_combination.name}.pkl",
        )
    )

