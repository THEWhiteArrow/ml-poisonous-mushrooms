from dataclasses import dataclass
from typing import List, Literal

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from lib.models.ModelWrapper import ModelWrapper


@dataclass
class ModelManager:
    task: Literal["classification"]

    def get_model_wrappers(self, use_sv: bool = False) -> List[ModelWrapper]:
        # --- TODO ---
        # Add more taks types and models
        if self.task == "classification":
            models = [
                ModelWrapper(model=RidgeClassifier(), allow_strings=False),
                ModelWrapper(model=RandomForestClassifier(), allow_strings=True),
                ModelWrapper(model=KNeighborsClassifier(), allow_strings=True),
                ModelWrapper(model=LGBMClassifier(), allow_strings=True),  # type: ignore
                ModelWrapper(model=XGBClassifier(), allow_strings=True),  # type: ignore
            ]

            if use_sv:
                models.append(ModelWrapper(model=SVC(), allow_strings=True))

            return models

        else:
            raise ValueError(
                "Bro. You had one job of selecting a proper task name and you failed..."
            )
