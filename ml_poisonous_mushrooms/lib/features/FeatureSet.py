from dataclasses import dataclass

from ml_poisonous_mushrooms.lib.features.FeatureCombination import FeatureCombination


@dataclass
class FeatureSet(FeatureCombination):
    is_optional: bool = True
