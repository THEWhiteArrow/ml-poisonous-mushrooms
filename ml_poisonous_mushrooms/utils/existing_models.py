import os
from typing import List

from ml_poisonous_mushrooms.utils.PathManager import PathManager
from ml_poisonous_mushrooms.utils.PrefixManager import PrefixManager


def get_existing_models(run_str: str) -> List[str]:
    existing_models = []

    model_run_dir_path = os.path.join(
        PathManager.MODELS_DIR_PATH.value,
        f"{PrefixManager.HYPER_OPT_PREFIX.value}{run_str}",
    )

    if not os.path.exists(model_run_dir_path):
        return existing_models

    for file in os.listdir(model_run_dir_path):
        if file.endswith(".pkl"):
            existing_models.append(file.split(".")[0])

    return existing_models
