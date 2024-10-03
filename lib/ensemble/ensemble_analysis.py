from pathlib import Path
from typing import Callable, Dict, Tuple, TypedDict

import pandas as pd
from sklearn.model_selection import cross_val_score

from lib.models.EnsembleModel import EnsembleModel
from lib.logger import setup_logger
from lib.pipelines.ProcessingPipelineWrapper import ProcessingPipelineWrapper

logger = setup_logger(__name__)


class EnsembleFunctionDto(TypedDict):
    load_data_func: Callable[[], Tuple[pd.DataFrame, pd.DataFrame]]
    engineer_features_func: Callable[[pd.DataFrame], pd.DataFrame]


def analyse_ensemble(model_run: str, ensemble_model: EnsembleModel, n_cv: int, ensemble_model_dir_path: Path, limit_data_percentage: float, function_dto: EnsembleFunctionDto) -> None:
    if limit_data_percentage < 0.0 or limit_data_percentage > 1.0:
        raise ValueError(
            f"Invalid limit data percentage value: {limit_data_percentage}"
        )

    results1 = results2 = {}

    logger.info("Testing all models from ensemble")
    results1 = verify_models_from_ensemble(ensemble_model, n_cv, limit_data_percentage, function_dto)

    logger.info("Testing ensemble model")
    results2 = verify_ensemble_model(ensemble_model, n_cv, limit_data_percentage, function_dto)

    final_results = {
        **results1,
        **results2,
    }

    logger.info("Saving results to csv file")
    df = pd.DataFrame(final_results.items(), columns=["Model", "Accuracy"])
    ensemble_analysis_results_path = ensemble_model_dir_path / \
        f"ensemble_analysis_{limit_data_percentage}_{model_run}.csv"
    df.to_csv(ensemble_analysis_results_path, index=False)

    logger.info(f"Results saved to csv file: {ensemble_analysis_results_path}")


def verify_models_from_ensemble(
    ensemble_model: EnsembleModel, n_cv: int, limit_data_percentage: float, function_dto: EnsembleFunctionDto
) -> Dict[str, float]:
    logger.info("Testing models from ensemble")

    train, test = function_dto["load_data_func"]()

    train = train.head(int(len(train) * limit_data_percentage))

    engineered_data = function_dto["engineer_features_func"](train).set_index("id")
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
    ensemble_model: EnsembleModel, n_cv: int, limit_data_percentage: float, function_dto: EnsembleFunctionDto
) -> Dict[str, float]:
    logger.info("Testing ensemble model")

    train, test = function_dto["load_data_func"]()

    train = train.head(int(len(train) * limit_data_percentage))

    engineered_data = function_dto["engineer_features_func"](train).set_index("id")

    X = engineered_data
    y = engineered_data["class"]
    scores = cross_val_score(
        estimator=ensemble_model, X=X, y=y, cv=n_cv, scoring="accuracy"
    )

    avg_acc = scores.mean()
    logger.info(f"Ensemble model has scored: {avg_acc}")

    return {"ensemble": avg_acc}
