from pathlib import Path
from typing import Callable, Dict, List, Tuple, TypedDict

import pandas as pd
from sklearn.model_selection import KFold

from lib.models.EnsembleModel import EnsembleModel
from lib.logger import setup_logger

logger = setup_logger(__name__)


class EnsembleFunctionDto(TypedDict):
    load_data_func: Callable[[], Tuple[pd.DataFrame, pd.DataFrame]]
    engineer_features_func: Callable[[pd.DataFrame], pd.DataFrame]


def analyse_ensemble(
    model_run: str,
    ensemble_model: EnsembleModel,
    n_cv: int,
    ensemble_model_dir_path: Path,
    limit_data_percentage: float,
    id_column: List[str] | str,
    function_dto: EnsembleFunctionDto,
) -> None:

    if limit_data_percentage < 0.0 or limit_data_percentage > 1.0:
        raise ValueError(
            f"Invalid limit data percentage value: {limit_data_percentage}"
        )

    logger.info("Loading data")

    train, test = function_dto["load_data_func"]()

    train = train.head(int(len(train) * limit_data_percentage))

    engineered_data = function_dto["engineer_features_func"](train).set_index(id_column)

    results: Dict[str, List[float]] = {
        model_name: [] for model_name in ensemble_model.combination_names
    }
    results["ensemble"] = []

    kfold = KFold(n_splits=n_cv)
    for i, (train_index, test_index) in enumerate(kfold.split(engineered_data)):
        X_train, X_test = (
            engineered_data.iloc[train_index],
            engineered_data.iloc[test_index],
        )
        y_train, y_test = X_train["class"], X_test["class"]

        logger.info(f"Testing model with fold {i + 1}")
        y_pred = ensemble_model.fit(X_train, y_train).predict(X_test)

        results["ensemble"].append((y_pred == y_test).sum() / len(y_test))
        for j, model in enumerate(ensemble_model.models):
            if ensemble_model.predictions is None:
                raise ValueError(
                    f"Model {ensemble_model.combination_names[j]} has not been fitted yet"
                )

            results[ensemble_model.combination_names[j]].append(
                (ensemble_model.predictions[j] == y_test).sum() / len(y_test)
            )

    logger.info("Saving results to csv file")
    logger.info(f"Results saved to csv file: {results}")

    df = pd.DataFrame(results)

    ensemble_analysis_results_path = (
        ensemble_model_dir_path
        / f"ensemble_analysis_{limit_data_percentage}_{model_run}.csv"
    )
    df.to_csv(ensemble_analysis_results_path, index=False)

    logger.info(f"Results saved to csv file: {ensemble_analysis_results_path}")
