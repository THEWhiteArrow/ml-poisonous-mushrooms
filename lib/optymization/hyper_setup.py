import datetime as dt
from pathlib import Path
from typing import Callable, List, Optional, Tuple, TypedDict

import pandas as pd

from lib.logger import setup_logger
from lib.models.HyperOptCombination import HyperOptCombination
from lib.optymization.optimization_study import CREATE_OBJECTIVE_TYPE
from lib.optymization.parrarel_optimization import run_parallel_optimization
from lib.utils.read_existing_models import read_existing_models

logger = setup_logger(__name__)


class HyperSetupDto(TypedDict):
    n_optimization_trials: int
    n_cv: int
    n_patience: int
    min_percentage_improvement: float
    model_run: Optional[str]
    limit_data_percentage: float
    processes: Optional[int]
    target_name: str
    id_column: str | List[str] | None
    output_dir_path: Path
    hyper_opt_prefix: str
    study_prefix: str


class HyperFunctionDto(TypedDict):
    create_objective_func: CREATE_OBJECTIVE_TYPE
    load_data_func: Callable[[], Tuple[pd.DataFrame, pd.DataFrame]]
    engineer_features_func: Callable[[pd.DataFrame], pd.DataFrame]
    create_combinations_func: Callable[[Optional[int]], List[HyperOptCombination]]


def setup_hyper(setup_dto: HyperSetupDto, function_dto: HyperFunctionDto):
    if setup_dto["model_run"] is None:
        model_run = dt.datetime.now().strftime("%Y%m%d%H%M")
    else:
        model_run = setup_dto["model_run"]

    if (
        setup_dto["limit_data_percentage"] < 0.0
        or setup_dto["limit_data_percentage"] > 1.0
    ):
        raise ValueError(
            "Invalid limit data percentage value: "
            + f"{setup_dto['limit_data_percentage']}"
        )

    logger.info(f"Starting hyper opt for run {setup_dto['model_run']}...")
    logger.info(f"Using {setup_dto['processes']} processes.")
    logger.info(f"Using {setup_dto['n_optimization_trials']} optimization trials.")
    logger.info(f"Using {setup_dto['n_cv']} cross validation.")
    logger.info(f"Using {setup_dto['n_patience']} patience.")
    logger.info(
        f"Using {setup_dto['min_percentage_improvement'] * 100}"
        + "% min percentage improvement."
    )
    logger.info(f"Using {setup_dto['limit_data_percentage'] * 100}% data")

    logger.info("Loading data...")
    train, test = function_dto["load_data_func"]()

    logger.info("Engineering features...")
    # --- NOTE ---
    # This could be a class that is injected with injector.
    all_data_size = len(train)
    limited_data_size = int(all_data_size * setup_dto["limit_data_percentage"])
    logger.info(f"Limiting data from {all_data_size} to {limited_data_size}")
    engineered_data = function_dto["engineer_features_func"](
        train.head(limited_data_size)
    )
    if setup_dto["id_column"] is not None:
        engineered_data = engineered_data.set_index(setup_dto["id_column"])

    all_model_combinations = function_dto["create_combinations_func"](
        setup_dto["processes"]
    )
    logger.info(f"Created {len(all_model_combinations)} combinations.")

    logger.info("Checking for existing models...")
    omit_names = read_existing_models(
        setup_dto["output_dir_path"]
        / f"{setup_dto['hyper_opt_prefix']}{setup_dto['model_run']}"
    )
    logger.info(f"Omitting {len(omit_names)} combinations.")

    # --- NOTE ---
    # Metadata is a dictionary that can be used to store any additional information
    metadata = {
        "limit_data_percentage": setup_dto["limit_data_percentage"],
    }

    logger.info("Starting parallel optimization...")
    _ = run_parallel_optimization(
        model_run=model_run,
        direction="maximize",
        all_model_combinations=all_model_combinations,
        X=engineered_data,
        y=engineered_data[setup_dto["target_name"]],
        n_optimization_trials=setup_dto["n_optimization_trials"],
        n_cv=setup_dto["n_cv"],
        n_patience=setup_dto["n_patience"],
        min_percentage_improvement=setup_dto["min_percentage_improvement"],
        output_dir_path=setup_dto["output_dir_path"],
        hyper_opt_prefix=setup_dto["hyper_opt_prefix"],
        study_prefix=setup_dto["study_prefix"],
        create_objective=function_dto["create_objective_func"],
        omit_names=omit_names,
        processes=setup_dto["processes"],
        metadata=metadata,
    )

    logger.info("Models has been pickled and saved to disk.")
