from dataclasses import dataclass
from typing import List, Literal

from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier


@dataclass
class ModelManager:
    task: Literal["classification"]

    def get_models(self, use_sv: bool = False) -> List[BaseEstimator]:
        # --- TODO ---
        # Add more taks types and models
        if self.task == "classification":
            models = [
                RidgeClassifier(),
                RandomForestClassifier(),
                KNeighborsClassifier(),
                LGBMClassifier(),  # type: ignore
                XGBClassifier(),  # type: ignore
            ]

            if use_sv:
                models.append(SVC())

            return models

        else:
            raise ValueError(
                "Bro. You had one job of selecting a proper task name and you failed..."
            )
