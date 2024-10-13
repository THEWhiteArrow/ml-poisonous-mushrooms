from dataclasses import dataclass
from pathlib import Path
import pickle
from typing import Callable, List, Tuple, cast

import pandas as pd

from lib.logger import setup_logger
from lib.models.EnsembleModel2 import EnsembleModel2


@dataclass
class PredictionSetupDto:
    ensemble_model_run: str
    output_dir_path: Path
    ensemble_prefix: str
    id_column: List[str] | str
    target_column: str


@dataclass
class PredictionFunctionDto:
    load_data_func: Callable[[], Tuple[pd.DataFrame, pd.DataFrame]]
    engineer_features_func: Callable[[pd.DataFrame], pd.DataFrame]


logger = setup_logger(__name__)


def setup_prediction(
    setup_dto: PredictionSetupDto, function_dto: PredictionFunctionDto
) -> pd.Series:

    logger.info(
        f"Prediction task started | ensemble_model_run: {setup_dto.ensemble_model_run}"
    )
    ensemble_model_file_name = f"ensemble_model_{setup_dto.ensemble_model_run}.ensemble"

    ensemble_model = cast(
        EnsembleModel2,
        pickle.load(
            open(
                setup_dto.output_dir_path
                / setup_dto.ensemble_prefix
                / ensemble_model_file_name,
                "rb",
            )
        ),
    )

    logger.info(f"Loaded ensemble model from {ensemble_model_file_name}")

    logger.info("Loading test data")
    train, test = function_dto.load_data_func()

    logger.info("Engineering features")
    engineered_data_train = function_dto.engineer_features_func(train).set_index(
        setup_dto.id_column
    )
    engineered_data_test = function_dto.engineer_features_func(test).set_index(
        setup_dto.id_column
    )

    X_train = engineered_data_train.drop(setup_dto.target_column, axis=1)
    y_train = engineered_data_train[setup_dto.target_column]

    X_test = engineered_data_test

    logger.info("Fitting ensemble model")
    ensemble_model.fit(X_train, y_train)

    logger.info("Predicting with ensemble model")
    y_pred = ensemble_model.predict(X_test)

    return y_pred
