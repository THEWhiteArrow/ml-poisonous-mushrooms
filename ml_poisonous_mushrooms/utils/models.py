from typing import Any, Callable, Dict, List, Literal, Optional
from dataclasses import dataclass, asdict
import pickle
import optuna
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
from ml_poisonous_mushrooms.utils.features import FeatureCombination, FeatureManager


def get_ridge_params(trial: optuna.Trial) -> Dict[str, Any]:
    return {
        "alpha": trial.suggest_float("alpha", 1e-3, 1000, log=True),
        "tol": trial.suggest_float("tol", 1e-5, 1e-1, log=True),
    }


def get_random_forest_params(trial: optuna.Trial) -> Dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 20),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
        "max_features": trial.suggest_categorical(
            "max_features", [None, "sqrt", "log2"]
        ),
    }


def get_kneighbors_params(trial: optuna.Trial) -> Dict[str, Any]:
    return {
        "n_neighbors": trial.suggest_int("n_neighbors", 3, 50),
        "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
        "p": trial.suggest_int("p", 1, 5),
    }


def get_svc_params(trial: optuna.Trial) -> Dict[str, Any]:
    return {
        "C": trial.suggest_float("C", 1e-3, 10, log=True),
        "kernel": trial.suggest_categorical(
            "kernel", ["linear", "poly", "rbf", "sigmoid"]
        ),
        "gamma": trial.suggest_categorical("gamma", ["scale", "auto"]),
        "tol": trial.suggest_float("tol", 1e-5, 1e-1, log=True),
    }


def get_lgbm_params(trial: optuna.Trial) -> Dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 20),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 1, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 2, 500),
        "min_child_samples": trial.suggest_int("min_child_samples", 3, 200),
        "subsample": trial.suggest_float("subsample", 0.1, 1),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 1),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
    }


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
        # Add more taks types and models
        if self.task == "classification":
            return [
                ModelWrapper(
                    model=RidgeClassifier(),
                    allow_strings=False,
                    get_params=get_ridge_params,
                ),
                ModelWrapper(
                    model=RandomForestClassifier(),
                    allow_strings=True,
                    get_params=get_random_forest_params,
                ),
                ModelWrapper(
                    model=KNeighborsClassifier(),
                    allow_strings=True,
                    get_params=get_kneighbors_params,
                ),
                ModelWrapper(
                    model=SVC(),
                    allow_strings=True,
                    get_params=get_svc_params
                ),
                ModelWrapper(
                    model=LGBMClassifier(),   # type: ignore
                    allow_strings=True,
                    get_params=get_lgbm_params
                )
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
                name=f"{model_wrapper.model.__class__.__name__}_{
                    combination.name}",
                feature_combination=combination,
            )
            for model_wrapper in self.model_wrappers
            for combination in all_feature_combinations
        ]

        return hyper_opt_model_combinations
