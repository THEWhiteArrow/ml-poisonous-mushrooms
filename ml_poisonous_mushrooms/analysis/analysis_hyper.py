from typing import Optional
import optuna
import pandas as pd

from lib.logger import setup_logger
from lib.optymization.results import load_hyper_opt_results, load_hyper_opt_studies
from ml_poisonous_mushrooms.utils.PrefixManager import PrefixManager
from ml_poisonous_mushrooms.utils.PathManager import PathManager

logger = setup_logger(__name__)


def analyse_hyper(
    model_run: Optional[str],
    display_plots: bool,
) -> None:
    if model_run is None:
        raise ValueError("Model run must be specified.")
    logger.info(f"Starting analysis for run {model_run}...")

    logger.info("Veryfing existing models...")
    hyper_opt_results = load_hyper_opt_results(
        model_run=model_run,
        output_dit_path=PathManager.OUTPUT_DIR_PATH.value,
        hyper_opt_prefix=PrefixManager.HYPER_OPT_PREFIX.value,
    )

    results_df = pd.DataFrame([result for result in hyper_opt_results])
    logger.info(f"Results data frame shape: {results_df.shape}")

    results_df.to_csv(
        PathManager.OUTPUT_DIR_PATH.value
        / f"{PrefixManager.HYPER_OPT_PREFIX.value}{model_run}"
        / f"results_{model_run}.csv",
        index=False,
    )

    if display_plots:
        hyper_opt_studies = load_hyper_opt_studies(
            model_run=model_run,
            output_dir_path=PathManager.OUTPUT_DIR_PATH.value,
            study_prefix=PrefixManager.STUDY_PREFIX.value,
        )
        for study in hyper_opt_studies:

            logger.info(
                f"Study {study.study_name}" +
                f" has {len(study.trials)} trials."
            )

            # optuna.visualization.plot_optimization_history(study).show()
            optuna.visualization.plot_slice(study).show()
            # optuna.visualization.plot_param_importances(study).show()


if __name__ == "__main__":
    logger.info("Starting hyper analysis...")
    analyse_hyper(model_run="202409260000", display_plots=True)
    logger.info("Analysis complete.")
