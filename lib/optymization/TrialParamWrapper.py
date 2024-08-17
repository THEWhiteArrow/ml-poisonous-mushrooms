from dataclasses import dataclass
from typing import Any, Dict

import optuna


@dataclass
class TrialParamWrapper:
    """A class that is to help the creation of the trial parameters."""

    def _get_ridge_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        return {
            "alpha": trial.suggest_float("alpha", 1e-3, 1000, log=True),
            "tol": trial.suggest_float("tol", 1e-5, 1e-1, log=True),
        }

    def _get_random_forest_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            "max_features": trial.suggest_categorical(
                "max_features", [None, "sqrt", "log2"]
            ),
        }

    def _get_kneighbors_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        return {
            "n_neighbors": trial.suggest_int("n_neighbors", 3, 50),
            "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
            "p": trial.suggest_int("p", 1, 5),
        }

    def _get_svc_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        return {
            "C": trial.suggest_float("C", 1e-3, 10, log=True),
            "kernel": trial.suggest_categorical(
                "kernel", ["linear", "poly", "rbf", "sigmoid"]
            ),
            "gamma": trial.suggest_categorical("gamma", ["scale", "auto"]),
            "tol": trial.suggest_float("tol", 1e-5, 1e-1, log=True),
        }

    def _get_lgbm_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 10, 63),
            "min_child_weight": trial.suggest_float(
                "min_child_weight", 1e-3, 100, log=True
            ),
            "subsample": trial.suggest_float("subsample", 0.1, 1),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 1),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 100, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 100, log=True),
            "gamma": trial.suggest_float("gamma", 1e-3, 10, log=True),
        }

    def get_params(self, model_name: str, trial: optuna.Trial) -> Dict[str, Any]:
        if model_name == "RidgeClassifier":
            return self._get_ridge_params(trial)
        elif model_name == "RandomForestClassifier":
            return self._get_random_forest_params(trial)
        elif model_name == "KNeighborsClassifier":
            return self._get_kneighbors_params(trial)
        elif model_name == "SVC":
            return self._get_svc_params(trial)
        elif model_name == "LGBMClassifier":
            return self._get_lgbm_params(trial)
        else:
            raise ValueError(f"Model {model_name} not supported.")
