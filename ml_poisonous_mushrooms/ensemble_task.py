import multiprocessing as mp
from typing import List

from lib.ensemble.ensemble_setup2 import (
    EnsembleFunctionDto2,
    EnsembleSetupDto2,
    setup_ensemble_v2,
)
from lib.logger import setup_logger
from ml_poisonous_mushrooms.data_load.data_load import load_data
from ml_poisonous_mushrooms.engineering.engineering_features import engineer_features
from ml_poisonous_mushrooms.utils.PrefixManager import PrefixManager
from ml_poisonous_mushrooms.utils.PathManager import PathManager


logger = setup_logger(__name__)


if __name__ == "__main__":
    logger.info("Starting ensemble task...")

    selected_model_names: List[str] = [
        "CatBoostClassifier_stem_cap_gill_ringandveil_other_top_0",
        "XGBClassifier_stem_cap_gill_ringandveil_other_top_1",
        "XGBClassifier_stem_cap_gill_ringandveil_other_top_2",
        "CatBoostClassifier_stem_cap_gill_ringandveil_other_top_1",
        "LGBMClassifier_stem_cap_gill_ringandveil_other_top_0",
        "LGBMClassifier_stem_cap_gill_ringandveil_other_top_1",
    ]

    function_dto = EnsembleFunctionDto2(
        load_data_func=load_data,
        engineer_features_func=engineer_features,
    )

    setup_dto = EnsembleSetupDto2(
        model_run="testing",
        meta_model=None,
        hyper_model_run="202410060000",
        selected_model_names=selected_model_names,
        limit_data_percentage=0.01,
        optimize=True,
        target_column="class",
        score_direction="maximize",
        id_column="id",
        task="classification",
        n_cv=5,
        processes=mp.cpu_count(),
        prediction_method="predict",
        # prediction_method="predict_proba",
        prediction_proba_target=None,
        output_dir_path=PathManager.OUTPUT_DIR_PATH.value,
        hyper_opt_prefix=PrefixManager.HYPER_OPT_PREFIX.value,
        ensemble_prefix=PrefixManager.ENSEMBLE_PREFIX.value,
    )

    setup_ensemble_v2(setup_dto=setup_dto, function_dto=function_dto)

    logger.info("Ensemble task complete.")
