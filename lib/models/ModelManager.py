from dataclasses import dataclass
from typing import List, Literal

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from lightgbm import LGBMClassifier

from lib.models.ModelWrapper import ModelWrapper


@dataclass
class ModelManager:
    task: Literal["classification"]

    def get_model_wrappers(self) -> List[ModelWrapper]:
        # --- TODO ---
        # Add more taks types and models
        if self.task == "classification":
            return [
                ModelWrapper(model=RidgeClassifier(), allow_strings=False),
                ModelWrapper(model=RandomForestClassifier(), allow_strings=True),
                ModelWrapper(model=KNeighborsClassifier(), allow_strings=True),
                ModelWrapper(model=SVC(), allow_strings=True),
                ModelWrapper(model=LGBMClassifier(), allow_strings=True),  # type: ignore
            ]

        else:
            raise ValueError(
                "Bro. You had one job of selecting a proper task name and you failed..."
            )
