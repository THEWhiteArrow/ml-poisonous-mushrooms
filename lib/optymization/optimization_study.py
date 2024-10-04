import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple
import gc

import optuna
import pandas as pd
import pickle

from lib.models.HyperOptCombination import HyperOptCombination
from lib.models.HyperOptResultDict import HyperOptResultDict
from lib.optymization.EarlyStoppingCallback import EarlyStoppingCallback
from lib.logger import setup_logger


logger = setup_logger(__name__)


CREATE_OBJECTIVE_TYPE = Callable[
    [pd.DataFrame, pd.DataFrame | pd.Series, HyperOptCombination, int],
    Callable[[optuna.Trial], float],
]


def get_existing_trials_info(
    trials: List[optuna.trial.FrozenTrial], min_percentage_improvement: float
) -> Tuple[int, float | None]:
    no_improvement_count = 0
    best_value = None

    for trial in trials:

        if best_value is None or (
            trial.value is not None
            and trial.value > best_value * (1.0 + min_percentage_improvement)
        ):
            best_value = trial.value
            no_improvement_count = 0
        elif trial.value is not None:
            no_improvement_count += 1
        elif trial.value is None:
            logger.warning(f"Trial {trial.number} has no value")

    return no_improvement_count, best_value


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
    create_objective_func: CREATE_OBJECTIVE_TYPE,
    metadata: Optional[Dict] = None,
) -> None:
    combination_name = model_combination.name
    logger.info(f"Optimizing model combination {i}: {combination_name}")

    X = X.copy()[model_combination.feature_combination.features]
    y = y.copy()

    os.makedirs(output_dir_path / f"{study_prefix}{model_run}", exist_ok=True)

    sql_path = Path(
        f"{output_dir_path}/{study_prefix}{model_run}/{combination_name}.db"
    )

    # Create an Optuna study for hyperparameter optimization
    study = optuna.create_study(
        direction=direction,
        study_name=f"{model_combination.name}",
        load_if_exists=True,
        storage=f"sqlite:///{sql_path}",
    )

    no_improvement_count, best_value = get_existing_trials_info(
        study.get_trials(), min_percentage_improvement
    )

    early_stopping = EarlyStoppingCallback(
        name=model_combination.name,
        patience=n_patience,
        min_percentage_improvement=min_percentage_improvement,
        best_value=best_value,
        no_improvement_count=no_improvement_count,
    )

    study.optimize(
        func=create_objective_func(X, y, model_combination, n_cv),
        n_trials=n_optimization_trials,
        callbacks=[early_stopping],  # type: ignore
    )

    # Save the best parameters and score
    best_params = study.best_params
    best_score = study.best_value
    # --- NOTE ---
    # The model is saved with the best parameters but it does not possess the correct weights
    best_model = model_combination.model.set_params(**best_params)

    logger.info(
        f"Model combination: {combination_name} "
        + f"has scored: {best_score} with params: {best_params}"
    )

    result = HyperOptResultDict(
        name=model_combination.name,
        score=best_score,
        params=best_params,
        model=best_model,
        features=model_combination.feature_combination.features,
        n_trials=len(study.trials),
        metadata=metadata,
    )

    os.makedirs(output_dir_path / f"{hyper_opt_prefix}{model_run}", exist_ok=True)

    try:
        results_path = Path(
            f"{output_dir_path}/{hyper_opt_prefix}{model_run}/{model_combination.name}.pkl"
        )

        pickle.dump(result, open(results_path, "wb"))
    except Exception as e:
        logger.error(f"Error saving model combination {combination_name}: {e}")
        raise e

    del X
    del y
    del study
    gc.collect()
