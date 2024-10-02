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
            "num_leaves": trial.suggest_int("num_leaves", 15, 50),
            "min_child_weight": trial.suggest_float(
                "min_child_weight", 1e-3, 100, log=True
            ),
            "subsample": trial.suggest_float("subsample", 0.1, 1),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 1),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 100, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 100, log=True),
        }

    def _get_xgb_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        return {
            "n_estimators": trial.suggest_int(
                "n_estimators", 100, 1200, step=50
            ),  # Number of trees in the ensemble
            "max_depth": trial.suggest_int(
                "max_depth", 3, 20
            ),  # Maximum depth of each tree
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.01, 0.3, log=True
            ),  # Learning rate
            "subsample": trial.suggest_float(
                "subsample", 0.5, 1.0
            ),  # Subsample ratio of the training instances
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree", 0.5, 1.0
            ),  # Subsample ratio of columns when constructing each tree
            "gamma": trial.suggest_float(
                "gamma", 0.01, 10.0, log=True
            ),  # Minimum loss reduction required to make a further partition on a leaf node of the tree
            "reg_alpha": trial.suggest_float(
                "reg_alpha", 1e-8, 100.0, log=True
            ),  # L1 regularization term on weights
            "reg_lambda": trial.suggest_float(
                "reg_lambda", 1e-8, 100.0, log=True
            ),  # L2 regularization term on weights
            "min_child_weight": trial.suggest_float(
                "min_child_weight", 1, 100, log=True
            ),  # Minimum sum of instance weight (hessian) needed in a child
        }

    def _get_catboost_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        return {
            # Number of boosting rounds
            "iterations": trial.suggest_int("iterations", 100, 1000),
            "depth": trial.suggest_int("depth", 4, 16),  # Depth of trees
            # Learning rate
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 1.0, log=True),
            # L2 regularization term
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True),
            # Bagging temperature
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
            # Number of splits for numerical features
            "border_count": trial.suggest_int("border_count", 32, 255),
            # Balancing of positive and negative weights
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 0.0, 10.0),
            # Randomness in tree-building process
            "random_strength": trial.suggest_float("random_strength", 0.0, 10.0),
            # Use one-hot encoding for features with max size
            "one_hot_max_size": trial.suggest_int("one_hot_max_size", 2, 10),
            # Random subspace method for feature selection
            "rsm": trial.suggest_float("rsm", 0.5, 1.0),
            # Overfitting detector type
            "od_type": trial.suggest_categorical("od_type", ["IncToDec", "Iter"]),
            # Number of iterations to wait for the overfitting detector
            "od_wait": trial.suggest_int("od_wait", 10, 50),
        }

    def get_params(self, model_name: str, trial: optuna.Trial) -> Dict[str, Any]:
        if "ridge" in model_name.lower():
            return self._get_ridge_params(trial)

        elif "randomforest" in model_name.lower():
            return self._get_random_forest_params(trial)

        elif "kneighbors" in model_name.lower():
            return self._get_kneighbors_params(trial)

        elif "svc" in model_name.lower():
            return self._get_svc_params(trial)

        elif "lgbm" in model_name.lower():
            return self._get_lgbm_params(trial)

        elif "xgb" in model_name.lower():
            return self._get_xgb_params(trial)

        elif "catboost" in model_name.lower():
            return self._get_catboost_params(trial)

        else:
            raise ValueError(f"Model {model_name} not supported.")
