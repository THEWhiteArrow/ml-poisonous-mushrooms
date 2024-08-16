import os
from typing import List

from models import HYPER_OPT_PREFIX, MODELS_DIR_PATH


def get_existing_models(run_str: str) -> List[str]:
    existing_models = []

    model_run_dir_path = os.path.join(MODELS_DIR_PATH, f"{HYPER_OPT_PREFIX}{run_str}")

    if not os.path.exists(model_run_dir_path):
        return existing_models

    for file in os.listdir(model_run_dir_path):
        if file.endswith(".pkl"):
            existing_models.append(file.split(".")[0])

    return existing_models
