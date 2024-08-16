from dataclasses import dataclass
from typing import List

from lib.features.FeatureCombination import FeatureCombination
from lib.features.FeatureManager import FeatureManager
from lib.models.HyperOptCombination import HyperOptCombination
from lib.models.ModelWrapper import ModelWrapper


@dataclass
class HyperOptManager:
    feature_manager: FeatureManager
    model_wrappers: List[ModelWrapper]

    def get_model_combinations(self) -> List[HyperOptCombination]:
        all_feature_combinations: List[FeatureCombination] = (
            self.feature_manager.get_all_possible_feature_combinations()
        )

        hyper_opt_model_combinations: List[HyperOptCombination] = [
            HyperOptCombination(
                model_wrapper=model_wrapper,
                name=f"{model_wrapper.model.__class__.__name__}_{
                    combination.name}",
                feature_combination=combination,
            )
            for model_wrapper in self.model_wrappers
            for combination in all_feature_combinations
        ]

        return hyper_opt_model_combinations

