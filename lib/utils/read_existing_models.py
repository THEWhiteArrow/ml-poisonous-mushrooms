import os
from pathlib import Path
from typing import List


def read_existing_models(path: Path) -> List[str]:
    existing_models = []

    if not os.path.exists(path):
        return existing_models

    for file in os.listdir(path):
        if file.endswith(".pkl"):
            existing_models.append(file.split(".")[0])

    return existing_models
