from dataclasses import asdict, dataclass
import os
import pickle
from typing import Any, Dict, List

from sklearn.base import BaseEstimator


@dataclass
class HyperOptResult:
    name: str
    score: float
    params: Dict[str, Any]
    model: BaseEstimator
    features: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def pickle_as_dict(self, path: str) -> None:
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))

        with open(path, "wb") as f:
            pickle.dump(self.to_dict(), f)

    def pickle(self, path: str) -> None:
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))

        with open(path, "wb") as f:
            pickle.dump(self, f)
