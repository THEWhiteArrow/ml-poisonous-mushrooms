from dataclasses import dataclass
from typing import List, TypedDict


@dataclass
class EnsembleConfigDict(TypedDict):
    model_run: str
    model_combination_names: List[str]
