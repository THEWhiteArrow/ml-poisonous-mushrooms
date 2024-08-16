from dataclasses import dataclass

from sklearn.base import BaseEstimator


@dataclass
class ModelWrapper:
    model: BaseEstimator
    allow_strings: bool
