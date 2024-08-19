from typing import Optional
import datetime as dt
import multiprocessing as mp

from lib.logger import setup_logger
from lib.optymization.parrarel_optimization import run_parallel_optimization
from ml_poisonous_mushrooms.data_load.data_load import load_data
from ml_poisonous_mushrooms.engineering.engineering_combinations import (
    create_combinations,
)
from ml_poisonous_mushrooms.engineering.engineering_features import engineer_features
from ml_poisonous_mushrooms.optimization.objective import create_objective
from ml_poisonous_mushrooms.utils.PrefixManager import PrefixManager
from ml_poisonous_mushrooms.utils.PathManager import PathManager
from ml_poisonous_mushrooms.utils.existing_models import get_existing_models

logger = setup_logger(__name__)


def hyper_opt(
    n_optimization_trials: int,
    n_cv: int,
    n_patience: int,
    model_run: Optional[str],
    limit_data_percentage: float,
    processes: Optional[int] = None,
):
    if model_run is None:
        model_run = dt.datetime.now().strftime("%Y%m%d%H%M")

    if limit_data_percentage < 0.0 or limit_data_percentage > 1.0:
        raise ValueError(
            f"Invalid limit data percentage value: {limit_data_percentage}"
        )

    logger.info(f"Starting hyper opt for run {model_run}...")
    logger.info(f"Using {processes} processes.")
    logger.info(f"Using {n_optimization_trials} optimization trials.")
    logger.info(f"Using {n_cv} cross validation.")
    logger.info(f"Using {n_patience} patience.")
    logger.info(f"Using {limit_data_percentage * 100}% data")

    logger.info("Loading data...")
    train, test = load_data()

    logger.info("Engineering features...")
    # --- NOTE ---
    # This could be a class that is injected with injector.
    all_data_size = len(train)
    limited_data_size = int(all_data_size * limit_data_percentage)
    logger.info(f"Limiting data from {all_data_size} to {limited_data_size}")
    engineered_data = engineer_features(train.head(limited_data_size)).set_index("id")

    all_model_combinations = create_combinations()
    logger.info(f"Created {len(all_model_combinations)} combinations.")

    logger.info("Checking for existing models...")
    omit_names = get_existing_models(model_run)
    logger.info(f"Omitting {len(omit_names)} combinations.")

    logger.info("Starting parallel optimization...")
    _ = run_parallel_optimization(
        model_run=model_run,
        direction="maximize",
        all_model_combinations=all_model_combinations,
        X=engineered_data,
        y=engineered_data["class"],
        n_optimization_trials=n_optimization_trials,
        n_cv=n_cv,
        n_patience=n_patience,
        output_dir_path=PathManager.OUTPUT_DIR_PATH.value,
        hyper_opt_prefix=PrefixManager.HYPER_OPT_PREFIX.value,
        study_prefix=PrefixManager.STUDY_PREFIX.value,
        create_objective=create_objective,
        omit_names=omit_names,
        processes=processes,
    )

    logger.info("Models has been pickled and saved to disk.")


if __name__ == "__main__":
    logger.info("Starting hyper opt script...")
    hyper_opt(
        n_optimization_trials=125,
        n_cv=5,
        n_patience=50,
        model_run=None,
        limit_data_percentage=1.0,
        processes=mp.cpu_count() // 2,
    )
    logger.info("Hyper opt script complete.")
