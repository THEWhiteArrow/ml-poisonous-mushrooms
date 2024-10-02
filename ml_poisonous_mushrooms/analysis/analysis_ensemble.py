from pathlib import Path

import pickle
from typing import Dict
import pandas as pd
from sklearn.model_selection import cross_val_score

from lib.models.EnsembleModel import EnsembleModel
from lib.logger import setup_logger
from lib.pipelines.ProcessingPipelineWrapper import ProcessingPipelineWrapper
from ml_poisonous_mushrooms.engineering.engineering_features import engineer_features
from ml_poisonous_mushrooms.data_load.data_load import load_data, load_ensemble_config
from ml_poisonous_mushrooms.utils.PrefixManager import PrefixManager
from ml_poisonous_mushrooms.utils.PathManager import PathManager

logger = setup_logger(__name__)


def analyse_ensemble(
    ensemble_model: EnsembleModel, n_cv: int, ensemble_model_dir_path: Path, limit_data_percentage: float = 1.0
) -> None:
    if limit_data_percentage < 0.0 or limit_data_percentage > 1.0:
        raise ValueError(
            f"Invalid limit data percentage value: {limit_data_percentage}"
        )
    results1 = results2 = {}

    logger.info("Testing all models from ensemble")
    results1 = verify_models_from_ensemble(
        ensemble_model, n_cv, limit_data_percentage)

    logger.info("Testing ensemble model")
    results2 = verify_ensemble_model(
        ensemble_model, n_cv, limit_data_percentage)

    final_results = {
        **results1,
        **results2,
    }

    logger.info("Saving results to csv file")
    df = pd.DataFrame(final_results.items(), columns=["Model", "Accuracy"])
    ensemble_analysis_results_path = ensemble_model_dir_path / \
        f"ensemble_analysis_results_{limit_data_percentage}.csv"
    df.to_csv(ensemble_analysis_results_path, index=False)

    logger.info(f"Results saved to csv file: {ensemble_analysis_results_path}")


def verify_models_from_ensemble(
    ensemble_model: EnsembleModel, n_cv: int, limit_data_percentage: float
) -> Dict[str, float]:
    logger.info("Testing models from ensemble")

    train, test = load_data()

    train = train.head(int(len(train) * limit_data_percentage))

    engineered_data = engineer_features(train).set_index("id")
    results = {}
    for i, model in enumerate(ensemble_model.models):
        logger.info(f"Testing model {ensemble_model.combination_names[i]}")
        X = engineered_data[ensemble_model.combination_feature_lists[i]]
        y = engineered_data["class"]
        pipeline = ProcessingPipelineWrapper().create_pipeline(model)

        scores = cross_val_score(
            estimator=pipeline, X=X, y=y, cv=n_cv, scoring="accuracy"
        )

        avg_acc = scores.mean()

        logger.info(
            f"Model {ensemble_model.combination_names[i]} has scored: {
                avg_acc}"
        )

        results[ensemble_model.combination_names[i]] = avg_acc

    return results


def verify_ensemble_model(
    ensemble_model: EnsembleModel, n_cv: int, limit_data_percentage: float
) -> Dict[str, float]:
    logger.info("Testing ensemble model")

    train, test = load_data()

    train = train.head(int(len(train) * limit_data_percentage))

    engineered_data = engineer_features(train).set_index("id")

    X = engineered_data
    y = engineered_data["class"]
    scores = cross_val_score(
        estimator=ensemble_model, X=X, y=y, cv=n_cv, scoring="accuracy"
    )

    avg_acc = scores.mean()
    logger.info(f"Ensemble model has scored: {avg_acc}")

    return {"ensemble": avg_acc}


if __name__ == "__main__":
    logger.info("Starting ensemble analysis script...")

    logger.info("Loading ensemble_config.json file")
    config = load_ensemble_config()

    logger.info("Loading ensemble model...")
    ensemble_model_dir_path = (
        PathManager.OUTPUT_DIR_PATH.value
        / f"{PrefixManager.ENSEMBLE_PREFIX.value}{config['model_run']}"
    )

    custom_suffix = "_01_3"
    ensemble_model_path = ensemble_model_dir_path / \
        f"ensemble_model_{config["model_run"]}{custom_suffix}.ensemble"

    ensemble_model = pickle.load(open(ensemble_model_path, "rb"))
    analyse_ensemble(
        ensemble_model=ensemble_model,
        n_cv=5,
        ensemble_model_dir_path=ensemble_model_dir_path,
        limit_data_percentage=0.5,
    )
    logger.info("Ensemble analysis script complete.")
