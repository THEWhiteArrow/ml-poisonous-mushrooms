import pickle
from typing import cast

import pandas as pd
from lib.logger import setup_logger
from lib.models.EnsembleModel import EnsembleModel
from ml_poisonous_mushrooms.data_load.data_load import load_data, load_ensemble_config
from ml_poisonous_mushrooms.engineering.engineering_features import engineer_features, unlabel_targets
from ml_poisonous_mushrooms.utils.PathManager import PathManager
from ml_poisonous_mushrooms.utils.PrefixManager import PrefixManager

logger = setup_logger(__name__)


if __name__ == '__main__':
    logger.info('Prediction task')

    config = load_ensemble_config()

    ensemble_model_dir_path = (
        PathManager.OUTPUT_DIR_PATH.value
        / f"{PrefixManager.ENSEMBLE_PREFIX.value}"
    )

    custom_suffix = ""
    ensemble_model_path = ensemble_model_dir_path / \
        f"ensemble_model_{config["model_run"]}{custom_suffix}.ensemble"

    ensemble_model = cast(EnsembleModel, pickle.load(
        open(ensemble_model_path, "rb")))

    logger.info(f"Loaded ensemble model from {ensemble_model_path}")

    # Load the test data
    train, test = load_data()

    train["train"] = 1
    test["train"] = 0

    data = pd.concat([train, test], axis=0)

    engineered_data = engineer_features(data).set_index("id")

    X_train = engineered_data.loc[engineered_data["train"] == 1]
    y_train = engineered_data.loc[engineered_data["train"] == 1, "class"]

    X_test = engineered_data.loc[engineered_data["train"] == 0]

    ensemble_model.fit(X_train, y_train)

    y_pred = ensemble_model.predict(X_test)

    y_pred_df = unlabel_targets(pd.DataFrame(
        y_pred, columns=["class"], index=X_test.index))

    y_pred_df.to_csv(
        ensemble_model_dir_path /
        f"predictions_{config['model_run']}{custom_suffix}.csv"
    )
