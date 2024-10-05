from pathlib import Path
from typing import List, TypedDict

from lib.ensemble.ensemble_analysis import EnsembleFunctionDto, analyse_ensemble
from lib.ensemble.ensemble_creation import create_ensemble_model_and_save
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

    ensemble_model = create_ensemble_model_and_save(
        model_run=setup_dto["hyper_model_run"],
        selected_model_names=setup_dto["selected_model_names"],
        hyper_models_dir_path=setup_dto["hyper_models_dir_path"],
        ensemble_model_dir_path=setup_dto["ensemble_model_dir_path"],
    )

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
