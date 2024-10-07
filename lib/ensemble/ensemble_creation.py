from pathlib import Path
import pickle
from typing import List, Tuple, cast

import pandas as pd

from lib.ensemble.ensemble_analysis import EnsembleFunctionDto
from lib.logger import setup_logger
from lib.models.EnsembleModel import EnsembleModel
from lib.models.HyperOptResultDict import HyperOptResultDict

logger = setup_logger(__name__)


def create_optimal_ensemble_model(
    selected_model_names: List[str],
    hyper_models_dir_path: Path,
    limit_data_percentage: float,
    n_cv: int,
    function_dto: EnsembleFunctionDto,
) -> Tuple[EnsembleModel, pd.DataFrame]:

    logger.info("Loading models from the config")

    hyper_opt_results: List[HyperOptResultDict] = []

    for model_name in selected_model_names:
        model_path = hyper_models_dir_path / f"{model_name}.pkl"

        model_data = cast(HyperOptResultDict, pickle.load(open(model_path, "rb")))
        model_data["name"] = model_name
        hyper_opt_results.append(model_data)

    logger.info("All models has been loaded")
    logger.info("Creating generic ensemble model")

    ensemble_model = EnsembleModel(
        models=[result["model"] for result in hyper_opt_results],
        combination_feature_lists=[result["features"] for result in hyper_opt_results],
        combination_names=[result["name"] for result in hyper_opt_results],
        scores=[result["score"] for result in hyper_opt_results],
    )

    logger.info("Loading data")
    train, test = function_dto["load_data_func"]()

    train = train.head(int(len(train) * limit_data_percentage))
    logger.info("Engineering features")
    engineered_data = function_dto["engineer_features_func"](train).set_index("id")

    logger.info("Optimizing ensemble model with bitmap")
    optimal_ensemble_model, optimization_df = ensemble_model.optimize(
        X=engineered_data,
        y=engineered_data["class"],
        n_cv=n_cv,
    )

    return optimal_ensemble_model, optimization_df
