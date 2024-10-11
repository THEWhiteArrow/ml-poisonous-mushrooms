import os
import pickle

from lib.ensemble.ensemble_creation import (
    EnsembleFunctionDto,
    EnsembleSetupDto,
    create_ensemble_model,
)
from lib.logger import setup_logger


logger = setup_logger(__name__)


def setup_ensemble(
    setup_dto: EnsembleSetupDto, function_dto: EnsembleFunctionDto
) -> None:
    logger.info("Starting ensemble model and analyze...")

    ensemble_model, optimization_df = create_ensemble_model(
        selected_model_names=setup_dto["selected_model_names"],
        hyper_models_dir_path=setup_dto["hyper_models_dir_path"],
        limit_data_percentage=setup_dto["limit_data_percentage"],
        n_cv=setup_dto["n_cv"],
        processes=setup_dto["processes"],
        optimize=setup_dto["optimize"],
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
        / f"ensemble_optimization_{setup_dto['limit_data_percentage']}_{setup_dto['hyper_model_run']}.csv",
        index=True,
    )
    logger.info("Optimization results have been saved.")
