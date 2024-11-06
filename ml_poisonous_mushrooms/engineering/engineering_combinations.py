from typing import List, Optional
from ml_poisonous_mushrooms.lib.features.FeatureManager import FeatureManager
from ml_poisonous_mushrooms.lib.features.FeatureSet import FeatureSet
from ml_poisonous_mushrooms.lib.models.HyperOptCombination import HyperOptCombination
from ml_poisonous_mushrooms.lib.models.HyperOptManager import HyperOptManager
from ml_poisonous_mushrooms.lib.models.ModelManager import ModelManager


def create_combinations(processes: Optional[int] = None) -> List[HyperOptCombination]:
    feature_manager = FeatureManager(
        feature_sets=[
            FeatureSet(
                name="stem",
                is_optional=False,
                features=[
                    "stem-height",
                    "stem-width",
                    "stem-root",
                    "stem-surface",
                    "stem-color",
                ],
            ),
            FeatureSet(
                name="cap",
                is_optional=False,
                features=[
                    "cap-diameter",
                    "cap-shape",
                    "cap-surface",
                    "cap-color",
                ],
            ),
            FeatureSet(
                name="gill",
                is_optional=False,
                features=[
                    "gill-spacing",
                    "gill-attachment",
                    "gill-color",
                ],
            ),
            FeatureSet(
                name="ringandveil",
                is_optional=False,
                features=[
                    "has-ring",
                    "ring-type",
                    "veil-type",
                    "veil-color",
                ],
            ),
            FeatureSet(
                name="other",
                is_optional=False,
                features=[
                    "spore-print-color",
                    "habitat",
                    "season",
                    "does-bruise-or-bleed",
                ],
            ),
        ]
    )

    model_manager = ModelManager(task="classification")

    hyper_manager = HyperOptManager(
        feature_manager=feature_manager,
        models=model_manager.get_models(
            use_additional=["adaboost"], processes=processes
        ),
    )

    return hyper_manager.get_model_combinations()
