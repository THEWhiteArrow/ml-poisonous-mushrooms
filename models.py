from dataclasses import dataclass, asdict
import pickle
from typing import List, Literal
from sklearn.base import BaseEstimator
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Ridge, RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier, XGBRegressor
from features import FeatureCombination, FeatureManager


@dataclass
class HyperOptModelCombination:
    model: BaseEstimator
    name: str
    feature_combination: FeatureCombination

    def pickle(self, path: str = "./") -> None:
        # Open the file in write-binary mode
        with open(f"{path}{self.name}_{self.feature_combination.name}.pkl", "wb") as f:
            pickle.dump(self, f)

    def pickle_as_dict(self, path: str = "./") -> None:
        # Create a dictionary from the dataclass fields
        data_dict = asdict(self)

        # Open the file in write-binary mode
        with open(
            f"{path}{self.name}_{self.feature_combination.name}_dict.pkl", "wb"
        ) as f:
            pickle.dump(data_dict, f)


@dataclass
class HyperOptManager:
    feature_manager: FeatureManager
    models: List[BaseEstimator]

    def get_all_model_combinations(self) -> List[HyperOptModelCombination]:
        all_feature_combinations: List[FeatureCombination] = (
            self.feature_manager.get_all_possible_feature_combinations()
        )

        hyper_opt_model_combinations: List[HyperOptModelCombination] = [
            HyperOptModelCombination(
                model=model,
                name=model.__class__.__name__,
                feature_combination=combination,
            )
            for model in self.models
            for combination in all_feature_combinations
        ]

        return hyper_opt_model_combinations


def get_models(task: Literal["regression", "classification"]) -> List[BaseEstimator]:
    if task == "regression":
        return [
            LogisticRegression(),
            Ridge(),
            # LightGBM(),
            HistGradientBoostingRegressor(),
            XGBRegressor(),
        ]
    elif task == "classification":
        return [
            RidgeClassifier(),
            # LightGBMClassifier(),
            XGBClassifier(),
            RandomForestClassifier(),
            KNeighborsClassifier(),
            SVC(),
        ]
    else:
        raise ValueError(
            "Bro. You had one job of selecting a proper task name and you failed..."
        )
