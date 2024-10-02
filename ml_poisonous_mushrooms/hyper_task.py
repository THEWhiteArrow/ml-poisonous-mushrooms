import multiprocessing as mp

from lib.optymization.hyper_setup import hyper_opt, HyperSetupDto, HyperFunctionDto
from lib.logger import setup_logger
from ml_poisonous_mushrooms.data_load.data_load import load_data
from ml_poisonous_mushrooms.engineering.engineering_combinations import (
    create_combinations,
)
from ml_poisonous_mushrooms.engineering.engineering_features import engineer_features
from ml_poisonous_mushrooms.optimization.objective import create_objective
from ml_poisonous_mushrooms.utils.PathManager import PathManager
from ml_poisonous_mushrooms.utils.PrefixManager import PrefixManager

logger = setup_logger(__name__)

if __name__ == "__main__":
    logger.info("Starting hyper_opt.py")

    function_dto = HyperFunctionDto(
        create_objective_func=create_objective,
        load_data_func=load_data,
        engineer_features_func=engineer_features,
        create_combinations_func=create_combinations,
    )

    setup_dto = HyperSetupDto(
        n_optimization_trials=125,
        n_cv=5,
        n_patience=5,
        min_percentage_improvement=0.1,
        model_run=None,
        limit_data_percentage=0.1,
        processes=mp.cpu_count(),
        target_name="class",
        id_column="id",
        output_dir_path=PathManager.OUTPUT_DIR_PATH.value,
        hyper_opt_prefix=PrefixManager.HYPER_OPT_PREFIX.value,
        study_prefix=PrefixManager.STUDY_PREFIX.value,
    )

    hyper_opt(setup_dto=setup_dto, function_dto=function_dto)

    logger.info("Finished hyper_opt.py")
