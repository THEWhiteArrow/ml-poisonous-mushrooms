from typing import Any, Dict, List, TypedDict
from dataclasses import dataclass

from sklearn.base import BaseEstimator


@dataclass
class HyperOptResultDict(TypedDict):
    name: str
    model: BaseEstimator
    features: List[str]
    params: Dict[str, Any]
    score: float
