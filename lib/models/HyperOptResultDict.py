from typing import Any, Dict, List, Optional, TypedDict
from dataclasses import dataclass

import optuna
from sklearn.base import BaseEstimator


@dataclass
class HyperOptResultDict(TypedDict):
    name: str
    model: BaseEstimator
    features: List[str]
    params: Dict[str, Any]
    score: float
    study: Optional[optuna.Study]
