from dataclasses import dataclass

from lib.features.FeatureCombination import FeatureCombination
from lib.models.ModelWrapper import ModelWrapper


@dataclass
class HyperOptCombination:
    model_wrapper: ModelWrapper
    name: str
    feature_combination: FeatureCombination
