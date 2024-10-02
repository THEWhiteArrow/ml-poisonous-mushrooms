from lib.ensemble.ensemble_creation import create_ensemble_model_and_save
from lib.logger import setup_logger
from ml_poisonous_mushrooms.data_load.data_load import load_ensemble_config
from ml_poisonous_mushrooms.utils.PrefixManager import PrefixManager
from ml_poisonous_mushrooms.utils.PathManager import PathManager

logger = setup_logger(__name__)


if __name__ == "__main__":
    logger.info("Starting ensemble script...")

    config = load_ensemble_config()

    hyper_models_dir_path = (
        PathManager.OUTPUT_DIR_PATH.value
        / f"{PrefixManager.HYPER_OPT_PREFIX.value}{config['model_run']}"
    )

    ensemble_model_dir_path = (
        PathManager.OUTPUT_DIR_PATH.value
        / f"{PrefixManager.ENSEMBLE_PREFIX.value}{config['model_run']}"
    )

    selected_model_names = config["model_combination_names"]
    model_run = config["model_run"]

    create_ensemble_model_and_save(
        model_run=model_run,
        selected_model_names=selected_model_names,
        hyper_models_dir_path=hyper_models_dir_path,
        ensemble_model_dir_path=ensemble_model_dir_path,
    )

    logger.info("Ensemble script complete.")
