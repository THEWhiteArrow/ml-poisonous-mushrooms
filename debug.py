

from sklearn.model_selection import cross_val_score
from data_load import load_data
from engineering_features import engineer_features
from features import FeatureManager, FeatureSet
from logger import logger
from models import HyperOptManager, ModelManager
from pipelines import ProcessingPipelineWrapper


def debug():
    logger.info("Loading data...")
    train, test = load_data()

    logger.info("Engineering data...")
    engineered_data = engineer_features(train.head(10000 * 1000)).set_index("id")

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
                is_optional=True,
                features=["gill-spacing", "gill-attachment", "gill-color"],
            ),
            FeatureSet(
                name="veil", is_optional=True, features=["veil-type", "veil-color"]
            ),
            FeatureSet(
                name="ring", is_optional=True, features=["has-ring", "ring-type"]
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
        model_wrappers=model_manager.get_model_wrappers(),
    )

    all_model_combinations = hyper_manager.get_model_combinations()

    choosen_combination = all_model_combinations[13]
    logger.info(choosen_combination.name)

    X = engineered_data.copy()[choosen_combination.feature_combination.features]
    y = engineered_data.copy()["class"]

    processing_pipeline_wrapper = ProcessingPipelineWrapper()
    model_wrapper = choosen_combination.model_wrapper
    model = model_wrapper.model

    allow_strings = model_wrapper.allow_strings

    logger.info(f"The model {model.__class__.__name__} {"allows strings" if allow_strings is True else "does NOT allow strings"}")
    pipeline = processing_pipeline_wrapper.create_pipeline(
        model=model, allow_strings=allow_strings
    )

    # processed_X = pipeline.fit_transform(X=X.copy())

    scores = cross_val_score(
        estimator=pipeline, X=X, y=y, cv=5, scoring="accuracy"
    )

    logger.info(f"Accuracy scores: {scores}")
    logger.info(f"Avg: {scores.mean()}")

    # --- TODO ---
    # The data that is processed includes the imputer fields and scaled fields
    # (it should impute the fields and them scaled them without creating new columns)
    logger.info("Data has been processed")

