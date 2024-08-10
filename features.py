from dataclasses import dataclass
import logging
from typing import List
import pandas as pd


@dataclass
class FeatureCombination:
    name: str
    features: List[str]


@dataclass
class FeatureSet(FeatureCombination):
    is_optional: bool


@dataclass
class FeatureManager:
    feature_sets: List[FeatureSet]

    def verify_features_existence(self, X: pd.DataFrame) -> bool:
        columns: List[str] = X.columns.to_list()

        all_features: List[str] = [
            feature
            for feature_group in self.feature_sets
            for feature in feature_group.features
        ]

        diff_feats: List[str] = list(set(all_features) - set(columns))

        if len(diff_feats) > 0:
            logging.warning(f"Missing features in the data: {diff_feats}")
            return False

        return True

    def get_all_possible_feature_combinations(self) -> List[FeatureCombination]:
        mandatory_feature_sets: List[FeatureSet] = [
            feat_set for feat_set in self.feature_sets if feat_set.is_optional is False
        ]
        optioanl_feature_sets: List[FeatureSet] = [
            feat_set for feat_set in self.feature_sets if feat_set.is_optional is True
        ]

        if len(optioanl_feature_sets) > 10:
            logging.warning(
                f"The number of optional feature sets is high - {len(optioanl_feature_sets)}"
            )

        bitmap = 2 ** len(optioanl_feature_sets) - 1
        possible_combinations: List[FeatureCombination] = []

        for i in range(bitmap + 1):
            combination_name: str = ""
            combination_features: List[str] = []

            for mandatory_set in mandatory_feature_sets:
                combination_name += f"{mandatory_set.name}_"
                combination_features.extend(mandatory_set.features)

            for j, optional_set in enumerate(optioanl_feature_sets):
                if i & (1 << j):
                    combination_name += f"{optional_set.name}_"
                    combination_features.extend(optional_set.features)

            possible_combinations.append(
                FeatureCombination(name=combination_name, features=combination_features)
            )

        return possible_combinations
