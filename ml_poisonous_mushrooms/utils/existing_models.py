import os
from typing import List


def get_existing_models(run_str: str) -> List[str]:
    existing_models = []

    dir_path = os.path.dirname(os.path.realpath(__file__))
    model_run_dir_path = os.path.join(dir_path, "models", f"hyper_opt_{run_str}")

    if not os.path.exists(model_run_dir_path):
        return existing_models

    for file in os.listdir(model_run_dir_path):
        if file.endswith(".pkl"):
            existing_models.append(file.split(".")[0])

    return existing_models
