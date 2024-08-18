from lib.features.FeatureManager import FeatureManager
from lib.features.FeatureSet import FeatureSet
from lib.models.HyperOptManager import HyperOptManager
from lib.models.ModelManager import ModelManager


def create_combinations():
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
                is_optional=True,
                features=["cap-diameter", "cap-shape", "cap-surface", "cap-color"],
            ),
            FeatureSet(
                name="gill",
                is_optional=False,
                features=["gill-spacing", "gill-attachment", "gill-color"],
            ),
            FeatureSet(
                name="ring_and_veil",
                is_optional=False,
                features=["has-ring", "ring-type", "veil-type", "veil-color"],
            ),
            FeatureSet(
                name="other",
                is_optional=True,
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
        models=model_manager.get_models(),
    )

    return hyper_manager.get_model_combinations()