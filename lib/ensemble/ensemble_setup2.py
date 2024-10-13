from dataclasses import dataclass
import os
from pathlib import Path
import pickle
from typing import List, Literal, Optional, Callable, Tuple

import pandas as pd
from sklearn.base import BaseEstimator

from lib.ensemble.ensemble_testing import optimize_ensemble, test_ensemble
from lib.logger import setup_logger
from lib.models.EnsembleModel2 import EnsembleModel2
from lib.utils.read_existing_models import read_hyper_results


logger = setup_logger(__name__)


@dataclass
class EnsembleFunctionDto2:
    load_data_func: Callable[[], Tuple[pd.DataFrame, pd.DataFrame]]
    engineer_features_func: Callable[[pd.DataFrame], pd.DataFrame]


@dataclass
class EnsembleSetupDto2:
    model_run: Optional[str]
    meta_model: Optional[BaseEstimator]
    hyper_model_run: str
    limit_data_percentage: float
    selected_model_names: List[str]
    optimize: bool
    score_direction: Literal["maximize", "minimize", "equal"]
    target_column: str
    id_column: List[str] | str
    prediction_method: Literal["predict", "predict_proba"]
    prediction_proba_target: Optional[str]
    task: Literal["classification", "regression"]
    n_cv: int
    processes: int
    output_dir_path: Path
    hyper_opt_prefix: str
    ensemble_prefix: str


def setup_ensemble_v2(
    setup_dto: EnsembleSetupDto2, function_dto: EnsembleFunctionDto2
) -> None:

    logger.info(f"Starting ensemble setup v2 | model_run: {setup_dto.model_run}")

    logger.info("Retriving hyperopt models...")
    hyper_model_results = read_hyper_results(
        path=setup_dto.output_dir_path
        / f"{setup_dto.hyper_opt_prefix}{setup_dto.hyper_model_run}",
        selection=setup_dto.selected_model_names,
    )

    logger.info("Loading data...")
    train, test = function_dto.load_data_func()

    logger.info(f"Limiting data to {setup_dto.limit_data_percentage * 100}%")
    train = train.sample(frac=setup_dto.limit_data_percentage)

    if setup_dto.score_direction == "maximize":
        score_sum = sum([results["score"] for results in hyper_model_results])
        weights = [results["score"] / score_sum for results in hyper_model_results]
    elif setup_dto.score_direction == "minimize":
        score_sum = sum([results["score"] for results in hyper_model_results])
        weights = [1 - results["score"] / score_sum for results in hyper_model_results]
    else:
        weights = None

    logger.info("Creating ensemble model...")
    ensemble_model = EnsembleModel2(
        models=[results["model"] for results in hyper_model_results],
        combination_features=[results["features"] for results in hyper_model_results],
        combination_names=setup_dto.selected_model_names,
        task=setup_dto.task,
        weights=weights,
        prediction_method=setup_dto.prediction_method,
        meta_model=setup_dto.meta_model,
    )

    logger.info("Engineering features...")
    data = function_dto.engineer_features_func(train).set_index(setup_dto.id_column)
    X = data.drop(setup_dto.target_column, axis=1)
    y = data[setup_dto.target_column]

    if (
        setup_dto.optimize
        and setup_dto.task == "classification"
        and setup_dto.prediction_method == "predict"
        and setup_dto.meta_model is None
    ):
        logger.info("Optimizing ensemble model...")
        best_combination_names, results_df = optimize_ensemble(
            ensemble_model, X, y, setup_dto.n_cv, setup_dto.processes
        )
    else:
        logger.info("Just testing ensemble model...")
        best_combination_names = setup_dto.selected_model_names
        results_df = test_ensemble(ensemble_model, X, y, setup_dto.n_cv)

    logger.info("Saving csv with results...")
    os.makedirs(setup_dto.output_dir_path, exist_ok=True)
    results_df.to_csv(
        setup_dto.output_dir_path
        / f"{setup_dto.ensemble_prefix}"
        / f"ensemble_results_{setup_dto.model_run}.csv",
        index=True,
    )

    logger.info("Creating final ensemble model...")
    final_ensemble_model = EnsembleModel2(
        models=[
            results["model"]
            for results in hyper_model_results
            if results["name"] in best_combination_names
        ],
        combination_features=[
            results["features"]
            for results in hyper_model_results
            if results["name"] in best_combination_names
        ],
        combination_names=best_combination_names,
        task=setup_dto.task,
        weights=weights,
        prediction_method=setup_dto.prediction_method,
        meta_model=setup_dto.meta_model,
    )

    logger.info("Saving final ensemble model...")
    os.makedirs(setup_dto.output_dir_path, exist_ok=True)
    pickle.dump(
        final_ensemble_model,
        open(
            setup_dto.output_dir_path
            / f"{setup_dto.ensemble_prefix}"
            / f"ensemble_model_{setup_dto.model_run}.ensemble",
            "wb",
        ),
    )

    logger.info("Ensemble setup v2 complete.")
