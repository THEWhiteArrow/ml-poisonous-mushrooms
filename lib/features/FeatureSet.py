from dataclasses import dataclass

from lib.features.FeatureCombination import FeatureCombination


@dataclass
class FeatureSet(FeatureCombination):
    is_optional: bool = True
