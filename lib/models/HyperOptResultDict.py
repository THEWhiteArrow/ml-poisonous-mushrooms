from typing import Any, Dict, List, Optional, TypedDict

from sklearn.base import BaseEstimator


class HyperOptResultDict(TypedDict):
    name: str
    model: BaseEstimator
    features: List[str]
    params: Dict[str, Any]
    score: float
    n_trials: int
    metadata: Optional[Dict]
