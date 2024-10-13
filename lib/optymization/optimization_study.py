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


def save_hyper_result(
    trail: optuna.trial.FrozenTrial,
    model_combination: HyperOptCombination,
    output_dir_path: Path,
    hyper_opt_prefix: str,
    model_run: str,
    study: optuna.study.Study,
    metadata: Optional[Dict] = None,
    suffix: str = "",
) -> None:
    best_params = trail.params
    best_score = trail.value
    best_model = model_combination.model.set_params(**best_params)

    result = HyperOptResultDict(
        name=model_combination.name,
        score=best_score,  # type: ignore
        params=best_params,
        model=best_model,
        features=model_combination.feature_combination.features,
        n_trials=len(study.trials),
        metadata=metadata,
    )

    os.makedirs(output_dir_path / f"{hyper_opt_prefix}{model_run}", exist_ok=True)

    try:
        results_path = Path(
            f"{output_dir_path}/{hyper_opt_prefix}{model_run}/{model_combination.name}{suffix}.pkl"
        )

        pickle.dump(result, open(results_path, "wb"))
    except Exception as e:
        logger.error(
            f"Error saving model combination {model_combination.name}{suffix}: {e}"
        )
        raise e


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

    X = X.copy()
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

    trials = study.get_trials()
    completed_trails = [
        trial
        for trial in trials
        if trial.state == optuna.trial.TrialState.COMPLETE and trial.value is not None
    ]

    sorted_trials = sorted(
        completed_trails, key=lambda trial: trial.value, reverse=direction == "maximize"  # type: ignore
    )

    top_trials = sorted_trials[:3]

    for i, trial in enumerate(top_trials):
        save_hyper_result(
            trail=trial,
            model_combination=model_combination,
            output_dir_path=output_dir_path,
            hyper_opt_prefix=hyper_opt_prefix,
            model_run=model_run,
            study=study,
            metadata=metadata,
            suffix=f"top_{i}",
        )

    del X
    del y
    del study
    gc.collect()
