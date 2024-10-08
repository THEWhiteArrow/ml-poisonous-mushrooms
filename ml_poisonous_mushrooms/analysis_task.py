from lib.logger import setup_logger
from lib.optymization.analysis_setup import setup_analysis
from ml_poisonous_mushrooms.utils.PrefixManager import PrefixManager
from ml_poisonous_mushrooms.utils.PathManager import PathManager

logger = setup_logger(__name__)


if __name__ == "__main__":
    logger.info("Starting hyper analysis...")

    display_plots = False
    setup_analysis(
        model_run="202410060000",
        output_dir_path=PathManager.OUTPUT_DIR_PATH.value,
        hyper_opt_prefix=PrefixManager.HYPER_OPT_PREFIX.value,
        study_prefix=PrefixManager.STUDY_PREFIX.value,
        display_plots=False,
    )
    logger.info("Analysis complete.")
