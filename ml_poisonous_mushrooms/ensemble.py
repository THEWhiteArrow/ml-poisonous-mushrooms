from pathlib import Path
from typing import List, cast
import pickle

from sklearn.model_selection import cross_val_score

from lib.models.EnsembleModel import EnsembleModel
from lib.models.HyperOptResultDict import HyperOptResultDict
from lib.logger import setup_logger
from lib.pipelines.ProcessingPipelineWrapper import ProcessingPipelineWrapper
from ml_poisonous_mushrooms.engineering.engineering_features import engineer_features
from ml_poisonous_mushrooms.utils.data_load import load_data, load_ensemble_config
from models import HYPER_OPT_PREFIX, ENSEMBLE_PREFIX

logger = setup_logger(__name__)


def create_ensemble_model_and_save() -> EnsembleModel:
    logger.info("Loading ensemble_config.json file")
    config = load_ensemble_config()

    logger.info("Loading models from the config")

    hyper_opt_results: List[HyperOptResultDict] = []

    for model_name in config["model_combination_names"]:
        model_path = Path(
            f"models/{HYPER_OPT_PREFIX}{config['model_run']}/{model_name}.pkl"
        )

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

    ensemble_model_path = Path(
        f"models/{ENSEMBLE_PREFIX}{config['model_run']}/ensemble_model.pkl"
    )

    logger.info(f"Saving ensemble model to {ensemble_model_path}")
    with open(ensemble_model_path, "wb") as f:
        pickle.dump(ensemble_model, f)

    return ensemble_model


def test_ensemble_model(ensemble_model: EnsembleModel, n_cv: int) -> float:
    logger.info("Testing ensemble model")
    # --- TODO ---
    # Add test code for the ensemble model
    train, test = load_data()

    engineered_data = engineer_features(train.head(1100 * 1000)).set_index("id")
    processing_pipeline_wrapper = ProcessingPipelineWrapper(pandas_output=False)
    pipeline = processing_pipeline_wrapper.create_pipeline(model=ensemble_model)

    X = engineered_data
    y = engineered_data["class"]
    scores = cross_val_score(estimator=pipeline, X=X, y=y, cv=n_cv, scoring="accuracy")

    avg_acc = scores.mean()

    logger.info(f"Ensemble model has scored: {avg_acc}")

    return avg_acc


if __name__ == "__main__":
    logger.info("Starting ensemble script...")
    ensamble_model = create_ensemble_model_and_save()

    logger.info("Testing ensemble model...")
    avg_acc = test_ensemble_model(ensemble_model=ensamble_model, n_cv=5)

    logger.info("Ensemble script complete.")
