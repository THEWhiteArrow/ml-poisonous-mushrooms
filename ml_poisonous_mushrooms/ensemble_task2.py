import datetime as dt
from pathlib import Path
import pickle
from typing import List, cast
from lightgbm import LGBMClassifier
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.ensemble import VotingClassifier, StackingClassifier

from lib.logger import setup_logger
from lib.models.HyperOptResultDict import HyperOptResultDict
from lib.pipelines.ProcessingPipelineWrapper import ProcessingPipelineWrapper
from ml_poisonous_mushrooms.data_load.data_load import load_data, load_ensemble_config
from ml_poisonous_mushrooms.engineering.engineering_features import engineer_features
from ml_poisonous_mushrooms.utils.PrefixManager import PrefixManager
from ml_poisonous_mushrooms.utils.PathManager import PathManager

logger = setup_logger(__name__)


def ensemble_v2(
    n_cv: int, hyper_models_dir_path: Path, selected_model_names: List[str]
) -> pd.DataFrame:

    logger.info("Loading models from the config")
    hyper_opt_results: List[HyperOptResultDict] = []
    for model_name in selected_model_names:
        model_path = hyper_models_dir_path / f"{model_name}.pkl"

        model_data = cast(HyperOptResultDict, pickle.load(open(model_path, "rb")))
        model_data["name"] = model_name
        hyper_opt_results.append(model_data)

    logger.info("Creating ensemble model")
    model_names: List[str] = []
    model_pipelines: List[Pipeline] = []
    model_scores: List[float] = []
    for model_data in hyper_opt_results:
        pipeline = ProcessingPipelineWrapper().create_pipeline(
            model=model_data["model"], features_in=model_data["features"]
        )

        model_names.append(model_data["name"].replace("__", "_"))
        model_pipelines.append(pipeline)
        model_scores.append(model_data["score"])

    model_weights: List[float] = [score / sum(model_scores) for score in model_scores]

    ensemble_voting_model = VotingClassifier(
        estimators=[
            (name, pipeline) for name, pipeline in zip(model_names, model_pipelines)
        ],
        weights=model_weights,
        voting="soft",
    )

    ensemble_stacking_model = StackingClassifier(
        estimators=[
            (name, pipeline) for name, pipeline in zip(model_names, model_pipelines)
        ],
        final_estimator=LGBMClassifier(n_jobs=-1, verbose=-1),  # type: ignore
        passthrough=True,
        n_jobs=-1,
        verbose=1,
    )

    train, test = load_data()
    train = train.head(int(len(train) * 0.1))
    data = engineer_features(train)

    X = data.drop(columns=["class"])
    y = data["class"]

    logger.info("CV voting ensemble")
    voting_scores = cross_val_score(
        ensemble_voting_model, X, y, cv=n_cv, n_jobs=-1, error_score="raise"
    )

    logger.info("CV stacking ensemble")
    stacking_scores = cross_val_score(
        ensemble_stacking_model, X, y, cv=n_cv, n_jobs=-1, error_score="raise"
    )

    results = pd.DataFrame(
        {
            "model": ["voting", "stacking"],
            "score": [np.mean(voting_scores), np.mean(stacking_scores)],
        }
    )

    return results


if __name__ == "__main__":
    model_run = dt.datetime.now().strftime("%Y%m%d%H%M")
    logger.info(f"Starting ensemble script {model_run}...")

    config = load_ensemble_config()

    hyper_models_dir_path = (
        PathManager.OUTPUT_DIR_PATH.value
        / f"{PrefixManager.HYPER_OPT_PREFIX.value}{config['model_run']}"
    )

    ensemble_model_dir_path = (
        PathManager.OUTPUT_DIR_PATH.value / PrefixManager.ENSEMBLE_PREFIX.value
    )

    selected_model_names = config["model_combination_names"]
    hyper_model_run = config["model_run"]

    # setup_ensemble(setup_dto=ensemble_setup_dto, function_dto=ensemble_function_dto)
    ensemble_results = ensemble_v2(
        n_cv=5,
        hyper_models_dir_path=hyper_models_dir_path,
        selected_model_names=selected_model_names,
    )

    ensemble_results.to_csv(
        PathManager.OUTPUT_DIR_PATH.value
        / "temp"
        / f"ensemble_v2_results_{hyper_model_run}.csv",
        index=False,
    )
    logger.info("Ensemble script complete.")
