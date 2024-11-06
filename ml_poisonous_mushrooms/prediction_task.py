from ml_poisonous_mushrooms.lib.logger import setup_logger
from ml_poisonous_mushrooms.lib.prediction.prediction_setup import (
    PredictionFunctionDto,
    PredictionSetupDto,
    setup_prediction,
)
from ml_poisonous_mushrooms.data_load.data_load import load_data
from ml_poisonous_mushrooms.engineering.engineering_features import (
    engineer_features,
    unlabel_targets,
)
from ml_poisonous_mushrooms.utils.PathManager import PathManager
from ml_poisonous_mushrooms.utils.PrefixManager import PrefixManager

logger = setup_logger(__name__)


if __name__ == "__main__":
    logger.info("Prediction task")

    prediction_setup_dto = PredictionSetupDto(
        ensemble_model_run="averaging",
        output_dir_path=PathManager.OUTPUT_DIR_PATH.value,
        ensemble_prefix=PrefixManager.ENSEMBLE_PREFIX.value,
        id_column="id",
        target_column="class",
    )

    PredictionFunctionDto = PredictionFunctionDto(
        load_data_func=load_data,
        engineer_features_func=engineer_features,
    )

    y_pred = setup_prediction(prediction_setup_dto, PredictionFunctionDto)

    y_pred_df = unlabel_targets(y_pred.rename("class").to_frame())

    y_pred_df.to_csv(
        PathManager.OUTPUT_DIR_PATH.value
        / PrefixManager.ENSEMBLE_PREFIX.value
        / f"predictions_{prediction_setup_dto.ensemble_model_run}.csv"
    )
