from typing import Any, Callable, Dict, List, Literal, Optional
from dataclasses import dataclass, asdict
import pickle
import optuna
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from features import FeatureCombination, FeatureManager


@dataclass
class ModelWrapper:
    model: BaseEstimator
    allow_strings: bool
    get_params: Callable[[optuna.Trial], Dict[str, Any]]


@dataclass
class ModelManager:
    task: Literal["classification"]

    def get_model_wrappers(self) -> List[ModelWrapper]:
        # --- TODO ---
        # Add more taks types and specify the params optimization
        if self.task == "classification":
            return [
                ModelWrapper(
                    model=RidgeClassifier(),
                    allow_strings=False,
                    get_params=lambda x: {},
                ),
                ModelWrapper(
                    model=RandomForestClassifier(),
                    allow_strings=True,
                    get_params=lambda x: {},
                ),
                ModelWrapper(
                    model=KNeighborsClassifier(),
                    allow_strings=True,
                    get_params=lambda x: {},
                ),
                ModelWrapper(
                    model=SVC(),
                    allow_strings=True,
                    get_params=lambda x: {},
                ),
            ]
        else:
            raise ValueError(
                "Bro. You had one job of selecting a proper task name and you failed..."
            )


@dataclass
class HyperOptModelCombination:
    model_wrapper: ModelWrapper
    name: str
    feature_combination: FeatureCombination
    score: Optional[float] = None
    hyper_parameters: Optional[Dict[str, Any]] = None

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
    model_wrappers: List[ModelWrapper]

    def get_model_combinations(self) -> List[HyperOptModelCombination]:
        all_feature_combinations: List[FeatureCombination] = (
            self.feature_manager.get_all_possible_feature_combinations()
        )

        hyper_opt_model_combinations: List[HyperOptModelCombination] = [
            HyperOptModelCombination(
                model_wrapper=model_wrapper,
                name=model_wrapper.model.__class__.__name__,
                feature_combination=combination,
            )
            for model_wrapper in self.model_wrappers
            for combination in all_feature_combinations
        ]

        return hyper_opt_model_combinations
