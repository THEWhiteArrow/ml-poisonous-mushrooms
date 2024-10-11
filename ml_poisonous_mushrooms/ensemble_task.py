from lib.ensemble.ensemble_creation import EnsembleFunctionDto
from lib.ensemble.ensemble_setup import EnsembleSetupDto, setup_ensemble
from lib.logger import setup_logger
from ml_poisonous_mushrooms.data_load.data_load import load_data, load_ensemble_config
from ml_poisonous_mushrooms.engineering.engineering_features import engineer_features
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
        PathManager.OUTPUT_DIR_PATH.value / PrefixManager.ENSEMBLE_PREFIX.value
    )

    selected_model_names = config["model_combination_names"]
    model_run = config["model_run"]

    ensemble_setup_dto = EnsembleSetupDto(
        hyper_models_dir_path=hyper_models_dir_path,
        ensemble_model_dir_path=ensemble_model_dir_path,
        selected_model_names=selected_model_names,
        hyper_model_run=model_run,
        n_cv=5,
        limit_data_percentage=1.0,
        id_column="id",
        processes=-1,
        optimize=True,
    )

    ensemble_function_dto = EnsembleFunctionDto(
        load_data_func=load_data,
        engineer_features_func=engineer_features,
    )

    setup_ensemble(setup_dto=ensemble_setup_dto, function_dto=ensemble_function_dto)

    logger.info("Ensemble script complete.")
