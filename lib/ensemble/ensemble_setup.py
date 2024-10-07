import os
from pathlib import Path
import pickle
from typing import List, TypedDict

from lib.ensemble.ensemble_analysis import EnsembleFunctionDto, analyse_ensemble
from lib.ensemble.ensemble_creation import create_optimal_ensemble_model
from lib.logger import setup_logger


logger = setup_logger(__name__)


class EnsembleSetupDto(TypedDict):
    hyper_models_dir_path: Path
    ensemble_model_dir_path: Path
    selected_model_names: List[str]
    hyper_model_run: str
    n_cv: int
    id_column: List[str] | str
    limit_data_percentage: float


def setup_ensemble(
    setup_dto: EnsembleSetupDto, function_dto: EnsembleFunctionDto
) -> None:
    logger.info("Starting ensemble model and analyze...")

    ensemble_model, optimization_df = create_optimal_ensemble_model(
        selected_model_names=setup_dto["selected_model_names"],
        hyper_models_dir_path=setup_dto["hyper_models_dir_path"],
        limit_data_percentage=setup_dto["limit_data_percentage"],
        n_cv=setup_dto["n_cv"],
        function_dto=function_dto,
    )

    logger.info("Saving ensemble model...")
    os.makedirs(setup_dto["ensemble_model_dir_path"], exist_ok=True)
    pickle.dump(
        ensemble_model,
        open(
            setup_dto["ensemble_model_dir_path"]
            / f"ensemble_model_{setup_dto['hyper_model_run']}.ensemble",
            "wb",
        ),
    )
    logger.info("Ensemble model has been saved.")
    logger.info("Saving optimization results...")
    optimization_df.to_csv(
        setup_dto["ensemble_model_dir_path"]
        / f"ensemble_optimization_{setup_dto['hyper_model_run']}.csv",
        index=False,
    )
    logger.info("Optimization results have been saved.")

    logger.info("Analyzing ensemble model...")
    analyse_ensemble(
        model_run=setup_dto["hyper_model_run"],
        ensemble_model=ensemble_model,
        n_cv=setup_dto["n_cv"],
        id_column=setup_dto["id_column"],
        ensemble_model_dir_path=setup_dto["ensemble_model_dir_path"],
        limit_data_percentage=setup_dto["limit_data_percentage"],
        function_dto=function_dto,
    )
