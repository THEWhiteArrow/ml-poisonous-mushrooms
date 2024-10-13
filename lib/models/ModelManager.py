from dataclasses import dataclass
from typing import List, Literal, Optional

from sklearn.base import BaseEstimator
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier


@dataclass
class ModelManager:
    task: Literal["classification"]

    def get_models(
        self, use_additional: List[str], processes: Optional[int] = None
    ) -> List[BaseEstimator]:
        # --- TODO ---
        # Investigate the speed of the KNeighborsClassifier, it seems that it is incredibly slow or something is
        # wrong with the implementation.

        job_count = processes if processes is not None else -1

        if self.task == "classification":
            models = [
                XGBClassifier(n_jobs=job_count),
                LGBMClassifier(n_jobs=job_count, verbosity=-1),
                RidgeClassifier(),
                CatBoostClassifier(verbose=False, thread_count=job_count),
            ]

            additional_models = [
                AdaBoostClassifier(algorithm="SAMME"),
                SVC(),
                RandomForestClassifier(n_jobs=job_count),
                KNeighborsClassifier(n_jobs=job_count, metric="cosine"),
            ]

            for model_name in use_additional:
                for model in additional_models:
                    if model_name.lower() in model.__class__.__name__.lower():
                        models.append(model)

            return models

        else:
            raise ValueError(
                "Bro. You had one job of selecting a proper task name and you failed..."
            )
