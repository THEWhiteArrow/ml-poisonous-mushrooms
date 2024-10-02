import os
from pathlib import Path
import pickle
from typing import List, cast

from lib.logger import setup_logger
from lib.models.EnsembleModel import EnsembleModel
from lib.models.HyperOptResultDict import HyperOptResultDict

logger = setup_logger(__name__)


def create_ensemble_model_and_save(
    model_run: str,
    selected_model_names: List[str],
    hyper_models_dir_path: Path,
    ensemble_model_dir_path: Path,
) -> None:

    logger.info("Loading models from the config")

    hyper_opt_results: List[HyperOptResultDict] = []

    for model_name in selected_model_names:
        model_path = hyper_models_dir_path / f"{model_name}.pkl"

        model_data = cast(HyperOptResultDict, pickle.load(open(model_path, "rb")))

        hyper_opt_results.append(model_data)

    logger.info("All models has been loaded")
    logger.info("Creating ensemble model")

    ensemble_model = EnsembleModel(
        models=[result["model"] for result in hyper_opt_results],
        combination_feature_lists=[result["features"] for result in hyper_opt_results],
        combination_names=[result["name"] for result in hyper_opt_results],
        scores=[result["score"] for result in hyper_opt_results],
    )

    logger.info("Ensemble model has been created")

    logger.info(f"Saving ensemble model to {ensemble_model_dir_path}")

    os.makedirs(ensemble_model_dir_path, exist_ok=True)

    pickle.dump(
        ensemble_model,
        open(ensemble_model_dir_path / f"ensemble_model_{model_run}.ensemble", "wb"),
    )
    logger.info("Ensemble model has been saved")
