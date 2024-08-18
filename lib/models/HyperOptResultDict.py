from typing import Any, Dict, List, TypedDict

from sklearn.base import BaseEstimator


class HyperOptResultDict(TypedDict):
    name: str
    score: float
    params: Dict[str, Any]
    model: BaseEstimator
    features: List[str]
