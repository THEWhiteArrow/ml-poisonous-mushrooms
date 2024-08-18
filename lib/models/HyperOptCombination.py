from dataclasses import dataclass

from sklearn.base import BaseEstimator

from lib.features.FeatureCombination import FeatureCombination


@dataclass
class HyperOptCombination:
    name: str
    model: BaseEstimator
    feature_combination: FeatureCombination
