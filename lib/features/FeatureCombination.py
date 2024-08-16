from dataclasses import dataclass
from typing import List


@dataclass
class FeatureCombination:
    name: str
    features: List[str]
